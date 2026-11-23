#!/usr/bin/env bash
# Wait for the main Mamba runs (5 scans x 3 seeds), then dump predictions and draw F7-F15.
#   bash phase2_t5/run_on_checkpoints.sh [MAMBA_DIR] [--once]
# MAMBA_DIR defaults to phase2_t5/outputs/t5-mamba-validation (copy Yerim's results there).
# Before processing, every run config is checked against run_t5.sh; a mismatch stops the script.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON:-python}"  # set PYTHON=/path/to/env/bin/python to override
mamba_dir="$root/outputs/t5-mamba-validation"
once=0
for arg in "$@"; do
  case "$arg" in
    --once) once=1 ;;
    *) mamba_dir="$(cd -- "$arg" && pwd)" ;;
  esac
done
interval="${INTERVAL:-120}"
cd "$root/../mamba_starter"

complete() {  # all five scans have seed_0..2/validation.json
  for scan in raster snake random blockwise spiral; do
    for seed in 0 1 2; do
      [[ -f "$mamba_dir/$scan/seed_$seed/validation.json" ]] || return 1
    done
  done
}

until complete; do
  if (( once )); then echo "Mamba runs incomplete in $mamba_dir"; exit 1; fi
  echo "$(date -Is) waiting for Mamba runs in $mamba_dir"
  sleep "$interval"
done
echo "$(date -Is) all Mamba runs found"

# Settings must match run_t5.sh so scans and seeds are comparable.
"$python" - "$mamba_dir" <<'EOF'
import json, sys
from pathlib import Path
expected = {"dataset": "fashion", "model": "mamba", "image_mode": "pixel", "image_size": 14,
            "image_filter": "t5_bilinear", "epochs": 20, "split_seed": 42, "permutation_seed": 42,
            "batch_size": 64, "d_model": 32, "d_state": 16, "n_layers": 2, "d_conv": 4,
            "lr": 0.001, "weight_decay": 0.01}
bad = []
for cfg_path in sorted(Path(sys.argv[1]).glob("*/config.json")):
    cfg = json.loads(cfg_path.read_text())
    diff = {k: (cfg.get(k), v) for k, v in expected.items() if cfg.get(k) != v}
    if diff:
        bad.append(f"{cfg_path.parent.name}: {diff}")
if bad:
    sys.exit("config differs from run_t5.sh (found, expected):\n" + "\n".join(bad))
print("configs match run_t5.sh")
EOF

process() {  # predictions for every finished run, then all model figures (F7-F17)
  local roots=("$mamba_dir" "$root/outputs/t5-mamba-validation-extra" "$root/outputs/t5-cnn-validation")
  [[ -d "$root/outputs/t5-ablation" ]] && roots+=("$root/outputs/t5-ablation")
  "$python" predict_t5.py --root "${roots[@]}" --out "$root/outputs/predictions" \
    --data_root "$root/data/FashionMNIST/raw"
  "$python" t5_figures.py --roots "${roots[@]}"
  "$python" t5_delta_analysis.py --roots "$mamba_dir" "$root/outputs/t5-mamba-validation-extra"
  echo "$(date -Is) figures updated in $root/figures"
}
process

# If the inward-spiral control was still training, redraw once it finishes (F15, F16b).
inward_done() {
  for seed in 0 1 2; do
    [[ -f "$root/outputs/t5-mamba-validation-extra/spiral_inward_seed$seed/seed_$seed/validation.json" ]] || return 1
  done
}
if ! inward_done && pgrep -f "scan spiral_inward" >/dev/null; then
  echo "$(date -Is) waiting for inward-spiral runs"
  while ! inward_done && pgrep -f "scan spiral_inward" >/dev/null; do sleep "$interval"; done
  process
fi
