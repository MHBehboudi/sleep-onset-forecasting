"""Causal, parameter-free featurization of the streamed Muse signal.

Raw input is microvolts with shape (B, 4, S), where S is a multiple of 128 samples (whole seconds).
Channels are [TP9, AF7, AF8, TP10] at 128 Hz.

Second s (1-indexed) is described by the 2-second frame raw[(s-2)*128 : s*128]. For s = 1 the first
second is repeated. Packets are 5 s long, and packet k ends at second 5k of the *received* stream,
so everything here is relative to the first packet the model sees, not to the true start of the night.

All trailing statistics are computed with cumulative sums along the time axis. The same code serves
batch training (whole night at once) and packet-by-packet streaming, and the two agree to float precision.
"""
import numpy as np
import torch
from torch import nn

FS = 128
NB = 26                                   # log-spaced spectral bands per channel, 0.5-32 Hz
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
MIRROR = [3, 2, 1, 0]                     # TP9 <-> TP10, AF7 <-> AF8
PAIR_BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 16), (16, 30), (30, 45)]


def _band_matrix():
    freqs = np.arange(129) * 0.5
    edges = np.geomspace(0.5, 32, NB + 1)
    m = np.zeros((129, NB), np.float32)
    for b in range(NB):
        sel = (freqs >= edges[b] - 1e-6) & (freqs < edges[b + 1] - 1e-6)
        if not sel.any():
            sel = np.abs(freqs - edges[b]) == np.abs(freqs - edges[b]).min()
        m[sel, b] = 1. / sel.sum()
    return m


def _pair_band_matrix():
    freqs = np.arange(129) * 0.5
    m = np.zeros((129, len(PAIR_BANDS)), np.float32)
    for b, (lo, hi) in enumerate(PAIR_BANDS):
        m[(freqs >= lo) & (freqs < hi), b] = 1.
    return m


def trail(v, w, dim):
    """Causal box mean over the last w steps along dim (fewer steps at the start)."""
    n = v.shape[dim]
    cs = torch.cumsum(v, dim)
    zero = torch.zeros_like(cs.narrow(dim, 0, 1))
    cs = torch.cat((zero, cs), dim)
    idx = torch.arange(1, n + 1, device=v.device)
    lo = (idx - w).clamp_min(0)
    num = cs.index_select(dim, idx) - cs.index_select(dim, lo)
    shape = [1] * v.dim(); shape[dim] = n
    return num / (idx - lo).to(v.dtype).view(shape)


def since(v, dim):
    """Causal running mean since the first received step."""
    n = v.shape[dim]
    shape = [1] * v.dim(); shape[dim] = n
    return torch.cumsum(v, dim) / torch.arange(1, n + 1, device=v.device, dtype=v.dtype).view(shape)


class Featurizer(nn.Module):
    """Per-second measurements: spectrum, signal-quality probe, waveform and cross-channel spectra."""

    def __init__(self):
        super().__init__()
        self.register_buffer('hann', torch.hann_window(256, periodic=False))
        self.register_buffer('bands', torch.from_numpy(_band_matrix()))
        self.register_buffer('pbands', torch.from_numpy(_pair_band_matrix()))
        f = torch.arange(129) * 0.5
        self.register_buffer('line', (((f >= 48) & (f <= 52)) | ((f >= 58) & (f <= 62))).float())
        self.register_buffer('hf', ((f >= 30) & (f <= 63.5)).float())
        self.register_buffer('lf', ((f >= 1) & (f < 30)).float())
        self.register_buffer('tot', ((f >= 0.5) & (f <= 63.5)).float())

    @staticmethod
    def frames(raw, prev=None):
        """raw (B,4,S); prev: the 128 samples preceding raw, or None at stream start -> (B,4,S/128,256)."""
        if prev is None:
            prev = raw[..., :128]
        return torch.cat((prev, raw), -1).unfold(-1, 256, 128)

    def base(self, frames):
        """frames (B,4,N,256) raw uV, possibly with NaN/Inf -> dict of per-second measurements."""
        finite = torch.isfinite(frames)
        nonfinite = 1. - finite.float().mean(-1)
        x = torch.where(finite, frames, torch.zeros_like(frames))
        last = x[..., 128:]
        last = last - last.mean(-1, keepdim=True)
        logstd = torch.log(last.std(-1) + 0.1)
        xc = x - x.mean(-1, keepdim=True)
        X = torch.fft.rfft(xc * self.hann, dim=-1)
        P = (X.real ** 2 + X.imag ** 2) / 256.
        logpow = torch.log(P @ self.bands + 1e-2)
        # quality probe: line noise, high-frequency (EMG) ratio, flat/clipped samples, kurtosis, mirror correlation
        line = torch.log(P @ self.line + 1e-3) - torch.log(P @ self.tot + 1e-3)
        hf = torch.log(P @ self.hf + 1e-3) - torch.log(P @ self.lf + 1e-3)
        d = torch.diff(x, dim=-1).abs()
        flat = (d < 0.05).float().mean(-1)
        maxdiff = torch.log1p(d.amax(-1))
        clip = ((x - x.median(-1, keepdim=True).values).abs() > 500).float().mean(-1)
        m2 = (xc ** 2).mean(-1) + 1e-3
        kurt = torch.log1p((xc ** 4).mean(-1) / m2 ** 2)
        xn = xc / m2.sqrt()[..., None]
        mirror = (xn * xn[:, MIRROR]).mean(-1)
        others = (xn.sum(1, keepdim=True) - xn) / 3.
        corr_rest = (xn * others).mean(-1) / (others.pow(2).mean(-1).sqrt() + 1e-3)
        # cross-spectra for the six electrode pairs in six bands
        Xb = X.unsqueeze(-1) * self.pbands
        auto = (P.unsqueeze(-1) * self.pbands).sum(-2)
        cross = torch.stack([(Xb[:, i] * Xb[:, j].conj()).sum(-2) / 256. for i, j in PAIRS], 1)
        return dict(logpow=logpow, logstd=logstd, line=line, hf=hf, flat=flat, maxdiff=maxdiff,
                    clip=clip, kurt=kurt, mirror=mirror, rest=corr_rest, nonfinite=nonfinite,
                    auto=auto, cross_re=cross.real, cross_im=cross.imag, last=last)

    @staticmethod
    def derive(b):
        """Trailing statistics over the per-second axis. Returns (chan, q, wave, pair).

        chan (B,4,N,55): spectral shape (gain-invariant), its departure from the last 5 min, the spectral
                         level and its departures from the 5-min and since-start means.
        q    (B,4,N,12): signal-quality probe.
        wave (B,4,N,128): last second of signal, robust-scaled by the channel's running amplitude.
        pair (B,6,N,12): magnitude-squared coherence and real cross-spectrum per pair and band (5 s smoothed).
        """
        lp = b['logpow']; level = lp.mean(-1); shape = lp - level[..., None]
        ls = b['logstd']
        dev_shape = shape - trail(shape, 300, 2)
        lvl = torch.stack((level, level - trail(level, 300, 2), level - since(level, 2)), -1)
        q = torch.stack((b['flat'], b['clip'], b['nonfinite'], b['line'], b['hf'], b['kurt'], b['maxdiff'],
                         b['mirror'], b['rest'], ls, ls - trail(ls, 120, 2), ls - since(ls, 2)), -1)
        chan = torch.cat((shape, dev_shape, lvl), -1)
        scale = torch.exp(trail(ls, 120, 2))[..., None]
        wave = torch.asinh(b['last'] / scale.clamp_min(0.5) * 2.)
        a = b['auto']; cr, ci = b['cross_re'], b['cross_im']
        a5 = trail(a, 5, 2); r5 = trail(cr, 5, 2); i5 = trail(ci, 5, 2)
        den = torch.stack([(a5[:, i] * a5[:, j]).clamp_min(1e-6).sqrt() for i, j in PAIRS], 1)
        coh = (r5 ** 2 + i5 ** 2).sqrt() / den
        pair = torch.cat((coh, r5 / den), -1)
        return chan, q, wave, pair


class Standardizer(nn.Module):
    """Fixed z-scoring of derived inputs, fitted once on clean training nights."""

    def __init__(self, dims):
        super().__init__()
        for k, d in dims.items():
            self.register_buffer(k + '_mu', torch.zeros(d)); self.register_buffer(k + '_sd', torch.ones(d))

    def fit(self, k, v):
        flat = v.reshape(-1, v.shape[-1])
        flat = flat[torch.isfinite(flat).all(-1)]
        getattr(self, k + '_mu').copy_(flat.mean(0)); getattr(self, k + '_sd').copy_(flat.std(0).clamp_min(1e-3))

    def __call__(self, k, v):
        return ((v - getattr(self, k + '_mu')) / getattr(self, k + '_sd')).clamp(-8, 8)
