"""Explicit signal conventions. Array shape [...,time,channel], time axis -2."""
import numpy as np
from scipy import signal


def _factor(m):
    if not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("factor must be a positive integer")
    return int(m)


def downsample(x, factor, method="aa", axis=-2):
    """Naive decimation or zero-phase FIR polyphase filtering before decimation.

    AA: SciPy's Kaiser(beta=5) FIR, padtype='line'. This is offline preprocessing,
    not an online causal filter; transients at window boundaries remain possible.
    """
    factor = _factor(factor)
    x = np.asarray(x)
    if method not in {"naive", "aa"}:
        raise ValueError("method must be naive or aa")
    if factor == 1:
        return x.copy()
    if method == "naive":
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(None, None, factor)
        return x[tuple(sl)].copy()
    if x.shape[axis] < 2:
        raise ValueError("anti-aliasing requires at least two input samples")
    return signal.resample_poly(x, 1, factor, axis=axis, window=("kaiser", 5.0), padtype="line")


def upsample(x, factor, method="poly", axis=-2):
    """Back to factor*L samples. Linear uses times k/factor; final tail held.

    Polyphase interpolation is an anti-imaging filter; it cannot undo aliasing.
    """
    factor = _factor(factor)
    x = np.asarray(x)
    if method == "poly":
        return signal.resample_poly(x, factor, 1, axis=axis, window=("kaiser", 5.0), padtype="line")
    if method != "linear":
        raise ValueError("method must be linear or poly")
    return np.apply_along_axis(lambda v: np.interp(np.arange(len(v)*factor)/factor,
                                                  np.arange(len(v)), v), axis, x)


def alias_frequency(frequency, sample_rate):
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    return abs((frequency + sample_rate / 2) % sample_rate - sample_rate / 2)


def power_spectrum(x, fs, axis=-2, detrend=True):
    """One-sided PSD [units^2/Hz], periodic Hann, common normalization.

    Bin spacing is fs/N, irrespective of windowing. Average PSD (power), never
    complex Fourier coefficients, across examples. Channel averaging is optional.
    """
    if fs <= 0:
        raise ValueError("sample rate must be positive")
    return signal.periodogram(x, fs=fs, window="hann", detrend="constant" if detrend else False,
                              return_onesided=True, scaling="density", axis=axis)


def ema(x, a):
    """Zero initial state h[-1]=0; h[t]=a*h[t-1]+(1-a)*x[t]."""
    if not 0 <= a < 1:
        raise ValueError("EMA requires 0 <= a < 1")
    return signal.lfilter([1-a], [1, -a], np.asarray(x), axis=-2)


def ema_features(x, alphas=(0.1, 0.5, 0.9, 0.99)):
    """All alphas x all channels x (mean, population std, RMS)."""
    if np.asarray(x).ndim != 3 or not alphas:
        raise ValueError("expected [examples,time,channels] and nonempty alphas")
    features = []
    for a in alphas:
        y = ema(x, a)
        features.extend([y.mean(1), y.std(1), np.sqrt(np.mean(y*y, axis=1))])
    return np.concatenate(features, axis=1)


def irregular_sample(x, keep_rate, rng):
    """Keep endpoints plus Bernoulli interior samples; return values, ticks, gaps.

    First gap=1 (zero state prior to the first observation). Other gaps are index
    differences in units of original sample intervals, NOT seconds. Endpoint
    retention makes the realized keep rate slightly larger than keep_rate.
    """
    if not 0 < keep_rate <= 1 or x.ndim != 2 or len(x) < 2:
        raise ValueError("require 0 < keep_rate <= 1 and x [time>=2,channel]")
    keep = rng.random(len(x)) < keep_rate
    keep[[0, -1]] = True
    ticks = np.flatnonzero(keep)
    gaps = np.r_[1, np.diff(ticks)].astype(np.float32)
    return x[ticks], ticks, gaps


def interpolate_irregular(values, ticks, length):
    return np.stack([np.interp(np.arange(length), ticks, values[:, c])
                     for c in range(values.shape[1])], axis=-1).astype(np.float32)
