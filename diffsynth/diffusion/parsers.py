import argparse


def add_dataset_base_config(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    return parser

def add_image_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution.")
    return parser

def add_video_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    return parser

def add_model_config(parser: argparse.ArgumentParser):
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--fp8_models", default=None, help="Models with FP8 precision, comma-separated.")
    parser.add_argument("--offload_models", default=None, help="Models with offload, comma-separated. Only used in splited training.")
    return parser

def add_training_config(parser: argparse.ArgumentParser):
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    parser.add_argument("--loss_mask_weight", type=float, default=0.0, help="Weight for masked latent loss using loss_mask_video.")
    parser.add_argument("--loss_mask_weight_end", type=float, default=None, help="Final masked loss weight for decay; unset keeps it constant.")
    parser.add_argument("--loss_mask_hold_ratio", type=float, default=0.0, help="Fraction of steps to hold the start mask weight before decay.")
    parser.add_argument("--loss_mask_decay", type=str, default="linear", choices=("linear", "cosine"), help="Decay type for mask loss weight.")
    parser.add_argument("--loss_mask_skip_first_frame", default=False, action="store_true", help="Skip mask loss on the first video frame.")
    return parser

def add_output_config(parser: argparse.ArgumentParser):
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    return parser

def add_lora_config(parser: argparse.ArgumentParser):
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--preset_lora_path", type=str, default=None, help="Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model.")
    parser.add_argument("--preset_lora_model", type=str, default=None, help="Which model the preset LoRA is fused to.")
    return parser

def add_gradient_config(parser: argparse.ArgumentParser):
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    return parser

def add_eval_config(parser: argparse.ArgumentParser):
    parser.add_argument("--val_dataset_base_path", type=str, default=None, help="Base path of the validation dataset.")
    parser.add_argument("--val_dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the validation dataset.")
    parser.add_argument("--val_dataset_repeat", type=int, default=1, help="Number of times to repeat the validation dataset per epoch.")
    parser.add_argument("--val_dataset_num_workers", type=int, default=0, help="Number of workers for validation data loading.")
    parser.add_argument("--val_data_file_keys", type=str, default=None, help="Data file keys for validation metadata. Comma-separated.")
    parser.add_argument("--val_batch_size", type=int, default=None, help="Batch size for validation. Defaults to --batch_size when unset.")
    parser.add_argument("--eval_every_n_epochs", type=int, default=1, help="Run evaluation every N epochs when validation data is provided.")
    parser.add_argument("--eval_max_batches", type=int, default=None, help="Maximum validation batches per eval pass.")
    return parser

def add_augmentation_config(parser: argparse.ArgumentParser):
    parser.add_argument("--aug_enable", default=False, action="store_true", help="Enable video-level augmentation.")
    parser.add_argument("--aug_hflip_prob", type=float, default=0.0, help="Horizontal flip probability for video-level augmentation.")
    parser.add_argument("--aug_color_jitter_prob", type=float, default=0.0, help="Color jitter probability for video-level augmentation.")
    parser.add_argument("--aug_color_jitter_strength", type=float, default=0.2, help="Color jitter strength.")
    parser.add_argument("--aug_fog_prob", type=float, default=0.0, help="Random fog probability for video-level augmentation.")
    parser.add_argument("--aug_rain_prob", type=float, default=0.0, help="Random rain probability for video-level augmentation.")
    parser.add_argument("--aug_snow_prob", type=float, default=0.0, help="Random snow probability for video-level augmentation.")
    parser.add_argument("--aug_sunflare_prob", type=float, default=0.0, help="Random sunflare probability for video-level augmentation.")
    return parser

def add_general_config(parser: argparse.ArgumentParser):
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_eval_config(parser)
    parser = add_augmentation_config(parser)
    return parser
