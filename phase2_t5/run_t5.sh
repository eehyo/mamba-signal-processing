#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output="$root/outputs/t5-mamba-validation"
d_conv="${1:-4}"
case "$d_conv" in
  4) ;;
  1) output="${output}-dconv1" ;;
  *) echo "Usage: bash run_t5.sh [4|1]" >&2; exit 2 ;;
esac
data_root="${FASHION_DATA_ROOT:-$root/data/FashionMNIST/raw}"
mkdir -p "$output"
cd "$root/../mamba_starter"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for scan in raster snake random blockwise spiral; do
  date -Is
  echo "Starting $scan, seeds 0 1 2"
  python -u train.py --dataset fashion --model mamba --device cuda \
    --d_conv "$d_conv" --data_root "$data_root" --image_mode pixel --image_size 14 \
    --image_filter t5_bilinear --scan "$scan" --seeds 0 1 2 --epochs 20 \
    --split_seed 42 --permutation_seed 42 --out "$output/$scan" \
    >> "$output/$scan.log" 2>&1
  echo "Finished $scan"
done
