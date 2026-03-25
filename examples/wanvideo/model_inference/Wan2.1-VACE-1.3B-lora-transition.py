"""
Inference script for the transition generation LoRA.

Inputs:
  --reference_image  : composited frame (scene + ad already placed)
  --mask             : mask image/video where ad=black(0), background=white(255)
  --lora_path        : path to trained LoRA .safetensors checkpoint
  --output           : output video path (default: output.mp4)
  --seed             : random seed (default: 42)
"""

import argparse
import numpy as np
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


def load_first_frame(path):
    """Load first frame from an image or video file."""
    if path.endswith(('.png', '.jpg', '.jpeg')):
        return Image.open(path).convert('RGB')
    # Video: use imageio
    import imageio
    reader = imageio.get_reader(path)
    frame = Image.fromarray(reader.get_data(0))
    reader.close()
    return frame


def build_vace_video(reference_image, mask_image, num_frames=81):
    """
    Construct vace_video = reference_image * (1 - mask).
    Ad pixels are preserved, background is black.
    Repeated for num_frames identical frames.
    """
    ref = np.array(reference_image).astype(np.float32)
    mask = np.array(mask_image.convert('RGB')).astype(np.float32) / 255.0
    inverted = 1.0 - mask
    vace_frame = (ref * inverted).astype(np.uint8)
    vace_pil = Image.fromarray(vace_frame)
    return [vace_pil] * num_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_image', type=str, required=True,
                        help='Composited frame with ad already placed.')
    parser.add_argument('--mask', type=str, required=True,
                        help='Mask: ad=black(0), background=white(255).')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='Path to trained LoRA .safetensors.')
    parser.add_argument('--lora_alpha', type=float, default=1.0,
                        help='LoRA scale (default: 1.0).')
    parser.add_argument('--output', type=str, default='output.mp4')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--num_frames', type=int, default=81)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Load pipeline
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device='cuda',
        model_configs=[
            ModelConfig(model_id='Wan-AI/Wan2.1-VACE-1.3B', origin_file_pattern='diffusion_pytorch_model*.safetensors'),
            ModelConfig(model_id='Wan-AI/Wan2.1-VACE-1.3B', origin_file_pattern='models_t5_umt5-xxl-enc-bf16.pth'),
            ModelConfig(model_id='Wan-AI/Wan2.1-VACE-1.3B', origin_file_pattern='Wan2.1_VAE.pth'),
        ],
        tokenizer_config=ModelConfig(model_id='Wan-AI/Wan2.1-T2V-1.3B', origin_file_pattern='google/umt5-xxl/'),
    )

    # Load LoRA into the VACE component
    pipe.load_lora(pipe.vace, args.lora_path, alpha=args.lora_alpha)

    # Prepare inputs
    reference_image = load_first_frame(args.reference_image)
    reference_image = reference_image.resize((args.width, args.height))

    mask_image = load_first_frame(args.mask)
    mask_image = mask_image.resize((args.width, args.height))

    vace_video = build_vace_video(reference_image, mask_image, num_frames=args.num_frames)

    # Run inference
    video = pipe(
        prompt='',
        vace_video=vace_video,
        vace_video_mask=mask_image,
        vace_reference_image=reference_image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        tiled=True,
    )

    save_video(video, args.output, fps=25, quality=5)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
