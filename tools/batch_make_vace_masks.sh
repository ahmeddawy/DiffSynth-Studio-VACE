#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/batch_make_vace_masks.sh /path/to/parent [--force] [--] [extra args]

This scans each subfolder in the parent directory, finds:
  - with_insertion_<folder>.mp4 (GT)
  - <folder>.mp4 (raw)
and writes:
  - loss_mask_<folder>.mp4

Extra args are passed through to tools/make_vace_mask.py.
Examples:
  tools/batch_make_vace_masks.sh test_samples
  tools/batch_make_vace_masks.sh test_samples -- --blur 0.7 --percentile 97
  tools/batch_make_vace_masks.sh test_samples --force
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$1"
shift

force=0
extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      force=1
      shift
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ ! -d "$parent_dir" ]]; then
  echo "Not a directory: $parent_dir" >&2
  exit 1
fi

shopt -s nullglob

for dir in "$parent_dir"/*/; do
  folder="$(basename "$dir")"
  gt="$dir/with_insertion_${folder}.mp4"
  if [[ ! -f "$gt" ]]; then
    gt_candidates=("$dir"/with_insertion*.mp4)
    if [[ ${#gt_candidates[@]} -gt 0 ]]; then
      gt="${gt_candidates[0]}"
    else
      echo "Skip $folder: no with_insertion*.mp4" >&2
      continue
    fi
  fi

  raw="$dir/${folder}.mp4"
  if [[ ! -f "$raw" ]]; then
    raw=""
    for cand in "$dir"/*.mp4; do
      base="$(basename "$cand")"
      if [[ "$base" == with_insertion* ]] || [[ "$base" == loss_mask_* ]] || [[ "$base" == masks* ]]; then
        continue
      fi
      raw="$cand"
      break
    done
    if [[ -z "$raw" ]]; then
      echo "Skip $folder: no raw mp4 found" >&2
      continue
    fi
  fi

  out="$dir/loss_mask_${folder}.mp4"
  if [[ -f "$out" && $force -ne 1 ]]; then
    echo "Skip $folder: $out exists (use --force to overwrite)" >&2
    continue
  fi

  echo "Processing $folder"
  python "$script_dir/make_vace_mask.py" \
    --gt "$gt" \
    --raw "$raw" \
    --output "$out" \
    --strict \
    --clip_to_min \
    --resize_to_raw \
    --macro_block_size 1 \
    "${extra_args[@]}"
done
