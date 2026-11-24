#!/usr/bin/env bash
# Official test evaluation under FINAL_PROTOCOL.md (run once; see that document).
# Every checkpoint is evaluated with mamba_starter/evaluate.py --split test, which records the
# protocol SHA-256. Results: outputs/t5-test/<group>/<scan>/seed_<k>.json. Finished files are skipped.
# Usage: bash phase2_t5/run_t5_test.sh [parallel_jobs]
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON:-python}"
jobs="${1:-6}"
out="$root/outputs/t5-test"
protocol="$root/FINAL_PROTOCOL.md"
mkdir -p "$out"

list="$out/checkpoints.txt"
: > "$list"
add() {  # group scan seed checkpoint
  # finished runs only: validation.json is written after a seed's last epoch
  if [[ -f "$4" && -f "$(dirname "$4")/validation.json" ]]; then echo "$1 $2 $3 $4" >> "$list"; fi
}
for seed in 0 1 2; do
  for scan in raster snake random blockwise spiral; do
    add mamba "$scan" "$seed" "$root/checkpoints/$scan/seed_$seed/best.pt"
    add mamba_k1 "$scan" "$seed" "$root/checkpoints-dconv1/$scan/seed_$seed/best.pt"
  done
  add mamba spiral_inward "$seed" "$root/outputs/t5-mamba-validation-extra/spiral_inward_seed$seed/seed_$seed/best.pt"
  add mamba column "$seed" "$root/outputs/t5-mamba-validation-column/column_seed$seed/seed_$seed/best.pt"
  for scan in raster snake random blockwise spiral spiral_inward column; do
    add cnn "$scan" "$seed" "$root/outputs/t5-cnn-validation/$scan/seed_$seed/best.pt"
    add cnn60 "$scan" "$seed" "$root/outputs/t5-cnn-validation-e60/$scan/seed_$seed/best.pt"
  done
done
echo "$(date -Is) $(wc -l < "$list") checkpoints, protocol sha256 $(sha256sum "$protocol" | cut -c1-16)"

cd "$root/../mamba_starter"
export CUDA_VISIBLE_DEVICES=""
run_one() {
  read -r group scan seed ckpt <<< "$1"
  dest="$out/$group/$scan/seed_$seed.json"
  if [[ -f "$dest" ]]; then return 0; fi
  mkdir -p "$(dirname "$dest")"
  "$python" evaluate.py --checkpoint "$ckpt" --data_root "$root/data/FashionMNIST/raw" --split test \
    --final_protocol "$protocol" --threads 4 --out "$dest.tmp" > /dev/null
  mv "$dest.tmp" "$dest"
  echo "$(date -Is) done $group $scan seed $seed"
}
export -f run_one
export out root python protocol
xargs -P "$jobs" -I{} bash -c 'run_one "$@"' _ {} < "$list"
echo "$(date -Is) test evaluation finished"
