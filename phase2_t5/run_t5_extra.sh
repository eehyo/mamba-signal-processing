#!/usr/bin/env bash
# Extra Phase 2 T5 runs, same settings as run_t5.sh:
#   cnn      - 1D CNN baseline, 6 scans x seeds 0 1 2 (CPU, minutes)
#   cnn60    - supplementary CNN: up to 60 epochs, stop after 5 epochs without gain (CPU)
#   inward   - Mamba on the inward spiral, one process per seed (GPU, ~2 h each)
# Usage: bash phase2_t5/run_t5_extra.sh cnn|cnn60|inward
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON:-python}"  # set PYTHON=/path/to/env/bin/python to override
common=(--dataset fashion --data_root "$root/data/FashionMNIST/raw" --image_mode pixel
        --image_size 14 --image_filter t5_bilinear --epochs 20 --split_seed 42
        --permutation_seed 42)
cd "$root/../mamba_starter"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

case "${1:-}" in
  cnn)
    output="$root/outputs/t5-cnn-validation"
    mkdir -p "$output"
    for scan in raster snake random blockwise spiral spiral_inward; do
      if [[ -f "$output/$scan/summary.json" ]]; then echo "skip CNN $scan (done)"; continue; fi
      rm -rf "$output/$scan"   # partial run from an interrupted session
      echo "$(date -Is) Starting CNN $scan"
      CUDA_VISIBLE_DEVICES="" "$python" -u train.py "${common[@]}" --model cnn --device cpu --scan "$scan" \
        --seeds 0 1 2 --out "$output/$scan" > "$output/$scan.log" 2>&1
      echo "$(date -Is) Finished CNN $scan"
    done
    ;;
  cnn60)  # supplementary: CNN up to 60 epochs, stop after 5 epochs without val macro-F1 gain
    output="$root/outputs/t5-cnn-validation-e60"
    mkdir -p "$output"
    pids=()
    for scan in raster snake random blockwise spiral spiral_inward; do
      if [[ -f "$output/$scan/summary.json" ]]; then echo "skip CNN60 $scan (done)"; continue; fi
      rm -rf "$output/$scan"
      CUDA_VISIBLE_DEVICES="" "$python" -u train.py "${common[@]}" --model cnn --device cpu --scan "$scan" \
        --epochs 60 --patience 5 --seeds 0 1 2 --out "$output/$scan" > "$output/$scan.log" 2>&1 &
      pids+=($!)
      echo "$(date -Is) Started CNN60 $scan (pid ${pids[-1]})"
    done
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    echo "$(date -Is) CNN60 runs finished (status $status)"
    exit "$status"
    ;;
  inward)
    output="$root/outputs/t5-mamba-validation-extra"
    mkdir -p "$output"
    gpus=(${INWARD_GPUS:-0 1 0})
    pids=()
    for seed in 0 1 2; do
      run="$output/spiral_inward_seed$seed"
      if [[ -f "$run/seed_$seed/validation.json" ]]; then echo "skip inward seed $seed (done)"; continue; fi
      rm -rf "$run"   # partial run from an interrupted session
      CUDA_VISIBLE_DEVICES="${gpus[$seed]}" "$python" -u train.py "${common[@]}" --model mamba \
        --device cuda --scan spiral_inward --seeds "$seed" \
        --out "$output/spiral_inward_seed$seed" > "$output/spiral_inward_seed$seed.log" 2>&1 &
      pids+=($!)
      echo "$(date -Is) Started inward seed $seed on GPU ${gpus[$seed]} (pid ${pids[-1]})"
    done
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    echo "$(date -Is) Inward runs finished (status $status)"
    exit "$status"
    ;;
  *) echo "usage: $0 cnn|cnn60|inward" >&2; exit 2 ;;
esac
