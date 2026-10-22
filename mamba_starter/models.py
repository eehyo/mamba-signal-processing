"""Classification wrappers. All inputs [batch,time,channels]; logits [batch,class]."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from mamba_block import MambaBlock, RMSNorm, valid_mask, zero_pad


def masked_mean(x, mask):
    return zero_pad(x, mask).sum(1) / mask.sum(1, keepdim=True).to(x.dtype)


class ResidualBlock(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaBlock(d_model=d_model, **kwargs)

    def forward(self, x, dt_mult=None, mask=None):
        return zero_pad(x + self.mixer(self.norm(x), dt_mult=dt_mult, mask=mask), mask)


class MambaClassifier(nn.Module):
    def __init__(self, input_dim=9, num_classes=6, d_model=32, n_layers=2, **kwargs):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([ResidualBlock(d_model, **kwargs) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def set_delta_scale(self, s):
        for layer in self.layers:
            layer.mixer.set_delta_scale(s)
        return self

    def set_record(self, enabled=True, states=False):
        for layer in self.layers:
            layer.mixer.set_record(enabled, states=states)
        return self

    @property
    def records(self):
        return [layer.mixer.record for layer in self.layers]

    def forward(self, x, dt_mult=None, mask=None):
        mask = valid_mask(x, mask)
        x = zero_pad(self.input_proj(zero_pad(x, mask)), mask)
        for layer in self.layers:
            x = layer(x, dt_mult=dt_mult, mask=mask)
        return self.head(masked_mean(self.norm_f(x), mask))


class CNNClassifier(nn.Module):
    def __init__(self, input_dim=9, num_classes=6, d_model=32, n_layers=2):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(input_dim if i == 0 else d_model, d_model, 5)
                                   for i in range(n_layers)])
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, dt_mult=None, mask=None):
        if dt_mult is not None:
            raise ValueError("CNN has no delta; use the gap-channel or interpolation strategy")
        mask = valid_mask(x, mask)
        x = zero_pad(x, mask)
        for conv in self.convs:
            x = F.relu(conv(F.pad(x.transpose(1, 2), (4, 0))).transpose(1, 2))
            x = zero_pad(x, mask)
        return self.head(masked_mean(x, mask))


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=9, num_classes=6, d_model=32, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, d_model, n_layers, batch_first=True)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, dt_mult=None, mask=None):
        if dt_mult is not None:
            raise ValueError("LSTM has no delta; use the gap-channel or interpolation strategy")
        mask = valid_mask(x, mask)
        packed = pack_padded_sequence(zero_pad(x, mask), mask.sum(1).cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed, _ = self.lstm(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
        return self.head(masked_mean(x, mask))


def build_model(name, input_dim, num_classes, d_model=32, n_layers=2,
                d_state=16, selective=True, dt_min=0.001, dt_max=0.1, d_conv=4):
    common = dict(input_dim=input_dim, num_classes=num_classes, d_model=d_model, n_layers=n_layers)
    if name == "mamba":
        return MambaClassifier(**common, d_state=d_state, selective=selective,
                               dt_min=dt_min, dt_max=dt_max, d_conv=d_conv)
    if name not in {"cnn", "lstm"}:
        raise ValueError("name must be mamba, cnn, or lstm")
    return (CNNClassifier if name == "cnn" else LSTMClassifier)(**common)
