#!/usr/bin/env python3
"""
Generate per-frame object masks by differencing a GT video and a raw video.

Usage:
  python tools/make_vace_mask.py --gt gt.mp4 --raw raw.mp4 --output masks.mp4
  python tools/make_vace_mask.py --gt gt.mp4 --raw raw.mp4 --output masks_dir
"""

import argparse
import os
import sys

import numpy as np
import imageio
import imageio.v3 as iio

try:
    import cv2
except Exception:
    cv2 = None


VIDEO_EXTS = {".mp4", ".webm", ".gif", ".avi", ".mov", ".mkv"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate object masks from GT and raw videos.")
    parser.add_argument("--gt", required=True, help="Path to GT video (with insertion).")
    parser.add_argument("--raw", required=True, help="Path to raw video (no insertion).")
    parser.add_argument("--output", required=True, help="Output path: video file or directory for frames.")
    parser.add_argument("--threshold", type=float, default=0.08, help="Diff threshold in [0,1].")
    parser.add_argument("--percentile", type=float, default=None, help="Use percentile-based threshold (0-100).")
    parser.add_argument("--diff_mode", choices=["mean", "max"], default="mean", help="Diff reduction across channels.")
    parser.add_argument("--blur", type=float, default=0.0, help="Gaussian blur sigma for diff map.")
    parser.add_argument("--erode", type=int, default=0, help="Erode iterations.")
    parser.add_argument("--dilate", type=int, default=0, help="Dilate iterations.")
    parser.add_argument("--num_frames", type=int, default=None, help="Maximum frames to process.")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index.")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS. Defaults to input FPS.")
    parser.add_argument("--strict", action="store_true", help="Require GT and raw videos to have the same frame count.")
    parser.add_argument("--macro_block_size", type=int, default=1, help="FFmpeg macro block size. Use 1 to prevent resizing.")
    return parser.parse_args()


def normalize_frame(frame):
    return frame.astype(np.float32) / 255.0


def compute_mask(gt_frame, raw_frame, args):
    gt = normalize_frame(gt_frame)
    raw = normalize_frame(raw_frame)
    diff = np.abs(gt - raw)
    if args.diff_mode == "max":
        diff_map = diff.max(axis=2)
    else:
        diff_map = diff.mean(axis=2)

    if args.blur and args.blur > 0:
        if cv2 is None:
            print("Warning: cv2 not available; blur skipped.", file=sys.stderr)
        else:
            diff_map = cv2.GaussianBlur(diff_map, (0, 0), args.blur)

    if args.percentile is not None:
        thr = np.percentile(diff_map, args.percentile)
    else:
        thr = args.threshold

    mask = (diff_map >= thr).astype(np.uint8) * 255

    if (args.erode or args.dilate) and cv2 is None:
        print("Warning: cv2 not available; erode/dilate skipped.", file=sys.stderr)
        return mask

    if cv2 is not None:
        kernel = np.ones((3, 3), dtype=np.uint8)
        if args.erode and args.erode > 0:
            mask = cv2.erode(mask, kernel, iterations=args.erode)
        if args.dilate and args.dilate > 0:
            mask = cv2.dilate(mask, kernel, iterations=args.dilate)

    return mask


def to_rgb(mask):
    if mask.ndim == 2:
        return np.repeat(mask[:, :, None], 3, axis=2)
    return mask


def main():
    args = parse_args()

    if not os.path.exists(args.gt):
        raise SystemExit(f"Missing GT video: {args.gt}")
    if not os.path.exists(args.raw):
        raise SystemExit(f"Missing raw video: {args.raw}")

    output_ext = os.path.splitext(args.output)[1].lower()
    output_is_video = output_ext in VIDEO_EXTS
    if not output_is_video:
        os.makedirs(args.output, exist_ok=True)

    gt_reader = imageio.get_reader(args.gt)
    raw_reader = imageio.get_reader(args.raw)
    gt_meta = gt_reader.get_meta_data()
    raw_meta = raw_reader.get_meta_data()
    if args.strict:
        try:
            gt_count = gt_reader.count_frames()
            raw_count = raw_reader.count_frames()
        except Exception as exc:
            gt_reader.close()
            raw_reader.close()
            raise SystemExit(f"Cannot count frames for strict mode: {exc}")
        if gt_count != raw_count:
            gt_reader.close()
            raw_reader.close()
            raise SystemExit(f"Frame count mismatch: gt={gt_count} raw={raw_count}")
    fps = args.fps if args.fps is not None else gt_meta.get("fps", 15)
    if fps is None:
        fps = raw_meta.get("fps", 15)

    writer = None
    if output_is_video:
        writer = imageio.get_writer(args.output, fps=fps, macro_block_size=args.macro_block_size)

    processed = 0
    if args.strict:
        total = gt_reader.count_frames()
        end_frame = total if args.num_frames is None else min(total, args.start_frame + args.num_frames)
        for idx in range(args.start_frame, end_frame):
            gt_frame = gt_reader.get_data(idx)
            raw_frame = raw_reader.get_data(idx)
            if gt_frame.shape != raw_frame.shape:
                raise ValueError(f"Frame shape mismatch at index {idx}: {gt_frame.shape} vs {raw_frame.shape}")
            mask = compute_mask(gt_frame, raw_frame, args)
            if output_is_video:
                writer.append_data(to_rgb(mask))
            else:
                frame_path = os.path.join(args.output, f"mask_{idx:06d}.png")
                iio.imwrite(frame_path, mask)
            processed += 1
    else:
        gt_iter = iter(gt_reader)
        raw_iter = iter(raw_reader)
        for idx, (gt_frame, raw_frame) in enumerate(zip(gt_iter, raw_iter)):
            if idx < args.start_frame:
                continue
            if args.num_frames is not None and processed >= args.num_frames:
                break
            if gt_frame.shape != raw_frame.shape:
                raise ValueError(f"Frame shape mismatch at index {idx}: {gt_frame.shape} vs {raw_frame.shape}")
            mask = compute_mask(gt_frame, raw_frame, args)
            if output_is_video:
                writer.append_data(to_rgb(mask))
            else:
                frame_path = os.path.join(args.output, f"mask_{idx:06d}.png")
                iio.imwrite(frame_path, mask)
            processed += 1

    gt_reader.close()
    raw_reader.close()
    if writer is not None:
        writer.close()

    print(f"Done. Frames processed: {processed}")


if __name__ == "__main__":
    main()
