import argparse
import csv
import os
import numpy as np
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


def load_first_frame(path):
    if path.endswith(('.png', '.jpg', '.jpeg')):
        return Image.open(path).convert('RGB')
    import imageio
    reader = imageio.get_reader(path)
    frame = Image.fromarray(reader.get_data(0))
    reader.close()
    return frame


def build_vace_video(reference_image, mask_image, num_frames=81):
    """vace_video = reference_image * (1 - mask): ad pixels kept, background black."""
    ref = np.array(reference_image).astype(np.float32)
    mask = np.array(mask_image.convert('RGB')).astype(np.float32) / 255.0
    vace_frame = Image.fromarray((ref * (1.0 - mask)).astype(np.uint8))
    return [vace_frame] * num_frames


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_base_path', type=str, required=True,
                    help='Base path of the dataset (same as training).')
parser.add_argument('--eval_csv', type=str, required=True,
                    help='Path to metadata_eval.csv.')
parser.add_argument('--lora_path', type=str, default='models/train/Wan2.1-VACE-1.3B_lora_transition/best.safetensors')
parser.add_argument('--lora_alpha', type=float, default=1.0)
parser.add_argument('--output_dir', type=str, default='eval_outputs')
parser.add_argument('--height', type=int, default=480)
parser.add_argument('--width', type=int, default=832)
parser.add_argument('--num_frames', type=int, default=81)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)
pipe.load_lora(pipe.vace, args.lora_path, alpha=args.lora_alpha)

with open(args.eval_csv) as f:
    rows = list(csv.DictReader(f))

print(f"Running inference on {len(rows)} eval samples...")

for i, row in enumerate(rows):
    sample_name = row['video'].split('/')[0]
    output_path = os.path.join(args.output_dir, f"{sample_name}.mp4")

    if os.path.exists(output_path):
        print(f"[{i+1}/{len(rows)}] Skipping (exists): {sample_name}")
        continue

    print(f"[{i+1}/{len(rows)}] {sample_name}")

    ref_path  = os.path.join(args.dataset_base_path, row['vace_reference_image'].strip())
    mask_path = os.path.join(args.dataset_base_path, row['vace_video_mask'].strip())

    reference_image = load_first_frame(ref_path).resize((args.width, args.height))
    mask_image      = load_first_frame(mask_path).resize((args.width, args.height))
    vace_video      = build_vace_video(reference_image, mask_image, num_frames=args.num_frames)

    video = pipe(
        prompt="",
        vace_video=vace_video,
        vace_video_mask=[mask_image] * args.num_frames,
        vace_reference_image=reference_image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        tiled=True,
    )
    save_video(video, output_path, fps=25, quality=5)
    print(f"  Saved: {output_path}")

print(f"\nDone. Results in: {args.output_dir}")
