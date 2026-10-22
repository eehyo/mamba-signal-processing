"""Run from the project root: python -m examples.hooks"""
import torch
from models import MambaClassifier
from data import SequenceDataset, collate_sequences
from signal_utils import irregular_sample
import numpy as np

torch.manual_seed(0)
torch.set_num_threads(2)
model = MambaClassifier(input_dim=9, d_model=16, d_state=8, n_layers=2)
x = torch.randn(2, 128, 9)
model.set_delta_scale(2.0)  # Hypothesis for transfer from 50 Hz to 25 Hz.
model.set_record(True, states=True)
with torch.no_grad():
    logits = model(x[:, ::2])
print("logits", tuple(logits.shape))
for i, rec in enumerate(model.records):
    print("layer", i, {key: tuple(val.shape) for key, val in rec.items()})
    neg_delta_A = -rec["delta"].unsqueeze(-1) * rec["A"]
    half_life = np.log(2) / neg_delta_A.clamp_min(1e-12)
    print("per-update local half-life mean", float(half_life.mean()))

model.set_delta_scale(1.0)
dataset = SequenceDataset(x.numpy(), np.array([0,1]), keep_rate=0.6, gap_mode="delta")
batch = collate_sequences([dataset[0], dataset[1]])
with torch.no_grad():
    logits = model(batch["x"], dt_mult=batch["dt"], mask=batch["mask"])
print("irregular lengths", batch["mask"].sum(1).tolist(), "logits", tuple(logits.shape))
model.set_record(False)  # Release the last batch's CPU recordings.
