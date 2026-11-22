"""SENTINEL: a fault-aware, causal streaming network for time-to-N2 forecasting from 4-channel Muse EEG.

    5 s packet ──> per-second featurizer (no parameters)
                     ├─ spectral shape / level departures      (chan, 55 per channel)
                     ├─ signal-quality probe                    (q, 12 per channel)
                     ├─ robust-scaled 1 s waveform              (wave, 128 samples)
                     └─ pairwise coherence / cross-spectra      (pair, 12 per electrode pair)
                   │
                   ▼
    shared per-channel encoder + channel identity embedding  ──> e_c (64)
    reliability head r_c = sigmoid(MLP[q_c, e_c])  (supervised with the known corruption mask)
                   │
                   ▼
    reliability-gated channel-SET fusion: attention over channels with log r_c added to the logits,
    plus a pair block weighted by r_i * r_j  ──>  1 token per second (d = 128)
                   │
                   ▼
    causal dilated TCN over seconds (receptive field 64 s) ──> packet token (+ frozen FM packet term)
                   │
                   ▼
    slow memory over packets with multi-timescale trailing means, departure from the stream mean and
    log elapsed time since the first received packet:  2-layer GRU  *or*  selective-timescale memory
                   │
                   ▼
    heads: point countdown 600*sigmoid(.) (output) | 121-way hazard (aux) | post-onset N2 state (aux)

A dead or noisy electrode gets a low reliability, receives almost no attention weight, and the
remaining electrodes produce a token from the same distribution. That is what made the model
robust to the blind test set's channel faults, which had cost earlier fixed-layout models 10-75 s.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .features import PAIRS, Featurizer, Standardizer, since, trail

TCN_DIL = [1, 2, 4, 8, 16, 32]
RF = 1 + sum(TCN_DIL)                     # 64 s receptive field (kernel size 2)


class TCNBlock(nn.Module):
    """Pre-norm residual block: causal depthwise dilated convolution + MLP."""

    def __init__(self, d, dil, p, k=2):
        super().__init__()
        self.dil = dil; self.k = k
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, k, dilation=dil, groups=d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p), nn.Linear(2 * d, d))

    def forward(self, h):                            # (B,N,d)
        z = self.norm(h).transpose(1, 2)
        z = F.pad(z, ((self.k - 1) * self.dil, 0))  # left padding only: causal
        z = self.conv(z).transpose(1, 2)
        return h + self.mlp(F.gelu(z))


def affine_prefix(decay, write):
    """All prefix states of h[t] = decay[t] * h[t-1] + write[t] with a log-depth parallel scan."""
    step = 1
    while step < write.shape[1]:
        write = torch.cat((write[:, :step], write[:, step:] + decay[:, step:] * write[:, :-step]), 1)
        decay = torch.cat((decay[:, :step], decay[:, step:] * decay[:, :-step]), 1)
        step *= 2
    return write


class TimescaleMemory(nn.Module):
    """Selective multi-timescale memory: 8 groups of leaky integrators with learned time constants
    (initialised 8 s ... 900 s). Input-dependent gates control the decay speed and how much of the
    current evidence is written; writes are further scaled by packet coverage (how much of the packet
    came from reliable electrodes), so faulted stretches do not overwrite the history. The readout sees
    the normalised history, the current value's departure from it, and the accumulated evidence mass.
    Training uses the parallel scan; streaming uses the identical one-step update."""

    def __init__(self, width, dropout):
        super().__init__()
        self.groups = 8
        if width % self.groups:
            raise ValueError('memory width must be divisible by 8')
        self.norm = nn.LayerNorm(width)
        self.value = nn.Linear(width, width)
        self.gates = nn.Linear(width, 3 * self.groups)
        self.log_tau = nn.Parameter(torch.linspace(np.log(8.), np.log(900.), self.groups))
        self.mix = nn.Sequential(nn.Linear(2 * width + self.groups, width), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(width, width))

    def terms(self, x, coverage):
        normalized = self.norm(x)
        value = self.value(normalized).tanh().unflatten(-1, (self.groups, -1))
        speed, trust, read = self.gates(normalized).unflatten(-1, (3, self.groups)).unbind(-2)
        tau = self.log_tau.clamp(np.log(3.), np.log(1800.)).exp()
        log_decay = -5 * (.25 + 1.5 * speed.sigmoid()) / tau
        weight = -torch.expm1(log_decay) * trust.sigmoid() * coverage
        write = torch.cat((value * weight.unsqueeze(-1), weight.unsqueeze(-1)), -1)
        return value, read.sigmoid().unsqueeze(-1), log_decay.exp().unsqueeze(-1), write

    def decode(self, x, value, gate, state):
        mass = state[..., -1:]
        observed = state[..., :-1] / (.05 + mass)
        features = torch.cat(((gate * observed).flatten(-2), (value - observed).flatten(-2),
                              (mass * 20).log1p().squeeze(-1)), -1)
        return x + self.mix(features)

    def forward(self, x, coverage):
        value, gate, decay, write = self.terms(x, coverage)
        return self.decode(x, value, gate, affine_prefix(decay, write))

    def step(self, x, coverage, previous=None):
        value, gate, decay, write = self.terms(x, coverage)
        state = write if previous is None else decay * previous + write
        return self.decode(x, value, gate, state), state


class Sentinel(nn.Module):
    """One ensemble member.

    temporal='gru'       : 2-layer GRU slow memory (5 members of the final ensemble, with FM input).
    temporal='timescale' : selective-timescale memory with a delay-conditioning readout between its two
                           layers (2 members, no FM input; trained with different losses, see losses.py).
    """

    def __init__(self, d=128, slow=192, p=0.1, temporal='gru', fm=False, use_quality=True, use_pairs=True):
        super().__init__()
        if temporal not in ('gru', 'timescale'):
            raise ValueError(temporal)
        self.temporal, self.use_fm = temporal, fm
        self.use_quality, self.use_pairs = use_quality, use_pairs
        self.nc, self.nq = 55, 12
        self.rf = RF
        self.feat = Featurizer()
        self.std = Standardizer(dict(chan=self.nc, q=self.nq, pair=12))
        # per-channel encoder (weights shared across electrodes)
        self.wave = nn.Sequential(nn.Conv1d(1, 16, 7, 2, 3), nn.GELU(), nn.Conv1d(16, 32, 5, 2, 2), nn.GELU(),
                                  nn.Conv1d(32, 32, 3, 2, 1), nn.GELU())
        self.wave_out = nn.Linear(64, 32)
        self.chan_in = nn.Sequential(nn.Linear(self.nc + self.nq, 96), nn.GELU(), nn.Linear(96, 64))
        self.chan_mix = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 64))
        self.chan_emb = nn.Parameter(torch.zeros(4, 64))
        self.chan_norm = nn.LayerNorm(64)
        # reliability head and channel-set fusion
        self.rel = nn.Sequential(nn.Linear(self.nq + 64, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.constant_(self.rel[-1].bias, 3.)
        self.att_logit = nn.Linear(64, 2)
        self.att_val = nn.Linear(64, 128)
        self.pair_emb = nn.Parameter(torch.zeros(6, 16))
        self.pair_mlp = nn.Sequential(nn.Linear(12 + 16, 64), nn.GELU(), nn.Linear(64, 64))
        self.fuse = nn.Sequential(nn.Linear(128 + 64 + 1, d), nn.LayerNorm(d))
        # fast temporal stack (seconds) and packet token
        self.tcn = nn.ModuleList([TCNBlock(d, k, p) for k in TCN_DIL])
        self.packet = nn.Linear(2 * d, d)
        if fm:   # frozen foundation-model packet term, zero-initialised so training starts at the no-FM function
            self.fm_proj = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 128))
            self.fm_mix = nn.Sequential(nn.GELU(), nn.Linear(128, d))
            nn.init.zeros_(self.fm_mix[-1].weight); nn.init.zeros_(self.fm_mix[-1].bias)
        # slow memory (packets)
        self.slow_in = nn.Sequential(nn.Linear(5 * d + 1, slow), nn.GELU())
        if temporal == 'gru':
            self.gru = nn.GRU(slow, slow, 2, batch_first=True, dropout=p)
        else:
            self.memory = nn.ModuleList(TimescaleMemory(slow, p) for _ in range(2))
            with torch.random.fork_rng(devices=[]):
                self.delay_readout = nn.Sequential(nn.LayerNorm(slow), nn.Linear(slow, 1, bias=False))
                self.delay_condition = nn.Linear(1, slow, bias=False)
            nn.init.zeros_(self.delay_condition.weight)
        # heads
        self.out_norm = nn.LayerNorm(slow)
        self.point = nn.Sequential(nn.Linear(slow, 96), nn.GELU(), nn.Linear(96, 1))
        nn.init.constant_(self.point[-1].bias, -1.1)
        self.hazard = nn.Linear(slow, 121)
        self.state = nn.Linear(slow, 1)
        self.drop = nn.Dropout(p)

    # ---------------- per-second encoder ----------------
    def seconds(self, chan, q, wave, pair):
        """Derived inputs -> fused per-second token u (B,N,d) and per-channel reliability logits (B,4,N)."""
        B, C, N, _ = chan.shape
        cs, qs, ps = self.std('chan', chan), self.std('q', q), self.std('pair', pair)
        if not self.use_quality:
            qs = torch.zeros_like(qs)
        w = self.wave(wave.reshape(B * C * N, 1, 128))
        w = self.wave_out(torch.cat((w.mean(-1), w.amax(-1)), -1)).view(B, C, N, 32)
        e = self.chan_mix(torch.cat((self.chan_in(torch.cat((cs, qs), -1)), w), -1)) + self.chan_emb[None, :, None]
        e = self.chan_norm(e)                                              # (B,4,N,64)
        rel_logit = self.rel(torch.cat((qs, e.detach()), -1)).squeeze(-1)  # (B,4,N)
        r = torch.sigmoid(rel_logit) if self.use_quality else torch.ones_like(rel_logit)
        logits = self.att_logit(e) + torch.log(r + 1e-3)[..., None]        # 2 attention heads over channels
        alpha = torch.softmax(logits, 1)
        v = self.att_val(e).view(B, C, N, 2, 64)
        pooled = (alpha[..., None] * v).sum(1).reshape(B, N, 128)
        if self.use_pairs:
            pf = self.pair_mlp(torch.cat((ps, self.pair_emb[None, :, None].expand(B, 6, N, 16)), -1))
            pw = torch.stack([r[:, i] * r[:, j] for i, j in PAIRS], 1)[..., None]
            pv = (pw * pf).sum(1) / (pw.sum(1) + 1e-3)
            tot = torch.log(pw.sum(1) + 1e-3)
        else:
            pv = pooled.new_zeros(B, N, 64); tot = pooled.new_zeros(B, N, 1)
        u = self.fuse(torch.cat((pooled, pv, tot), -1))
        return self.drop(u), rel_logit

    def run_tcn(self, u):
        for blk in self.tcn:
            u = blk(u)
        return u

    def packets(self, z, fmtok=None):
        """z (B,N,d) per-second TCN output, N a multiple of 5 -> (B,K,d) packet tokens."""
        B, N, d = z.shape
        zz = z.view(B, N // 5, 5, d)
        p = self.packet(torch.cat((zz[:, :, -1], zz.mean(2)), -1))
        return p if fmtok is None else p + fmtok

    def fm_from_embed(self, e, rel):
        """e (B,K,4,512) shared frozen embeddings -> member-specific, reliability-pooled packet term."""
        B, K, C, _ = e.shape
        proj = self.fm_proj(e)
        r = torch.sigmoid(rel).view(B, C, K, 5).mean(-1).permute(0, 2, 1) if self.use_quality else proj.new_ones(B, K, C)
        w = r / (r.sum(-1, keepdim=True) + 1e-3)
        return self.fm_mix((w[..., None] * proj).sum(2))

    # ---------------- slow memory and heads ----------------
    def slow_inputs(self, p):
        """p (B,K,d) -> slow-memory input with 1/4/16-minute trailing means, departure from the stream mean,
        and log elapsed time since the first received packet."""
        K = p.shape[1]
        t = torch.arange(1, K + 1, device=p.device, dtype=p.dtype) * 5.
        el = torch.log1p(t / 60.).view(1, K, 1).expand(p.shape[0], K, 1)
        return self.slow_in(torch.cat((p, trail(p, 12, 1), trail(p, 48, 1), trail(p, 192, 1), p - since(p, 1), el), -1))

    def heads(self, h):
        h = self.out_norm(h)
        self.last_h = h
        return 600. * torch.sigmoid(self.point(h).squeeze(-1)), self.hazard(h), self.state(h).squeeze(-1)

    def packet_coverage(self, rel, base):
        """Fraction of each packet carried by present, reliable electrodes (timescale memory write gate)."""
        present = (base['nonfinite'] < .1) & (base['flat'] < .98)
        confidence = rel.detach().sigmoid() * present
        return confidence.mean(1).unflatten(1, (-1, 5)).mean(-1, keepdim=True)

    def condition_delay(self, first):
        score = self.delay_readout(first)
        return first + self.delay_condition((score / 2).tanh()), score.squeeze(-1)

    def memory_forward(self, s, coverage=None):
        if self.temporal == 'gru':
            self.gru.flatten_parameters()
            return self.gru(s)[0]
        first = self.memory[0](s, coverage)
        conditioned, self.delay_score = self.condition_delay(first)
        return self.memory[1](conditioned, coverage)

    def forward(self, raw, fm_embed=None):
        """raw (B,4,640*K) uV, whole night prefix. fm_embed: optional (B,K,4,512) frozen FM embeddings.

        Returns per-packet (point countdown s, hazard logits, post-onset state logit) and per-second
        reliability logits (B,4,5K)."""
        b = self.feat.base(self.feat.frames(raw))
        chan, q, wave, pair = self.feat.derive(b)
        u, rel = self.seconds(chan, q, wave, pair)
        fmtok = None
        if self.use_fm:
            if fm_embed is None:
                raise ValueError('this member needs frozen FM packet embeddings (see fm.embed_packets)')
            fmtok = self.fm_from_embed(fm_embed, rel)
        pk = self.packets(self.run_tcn(u), fmtok)
        coverage = self.packet_coverage(rel, b) if self.temporal == 'timescale' else None
        h = self.memory_forward(self.slow_inputs(pk), coverage)
        point, hz, st = self.heads(h)
        return point, hz, st, rel
