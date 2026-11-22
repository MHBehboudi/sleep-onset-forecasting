"""Packet-by-packet inference that reproduces the batch forward pass on the received prefix.

The evaluator calls predict(x) once per 5 s packet (x: 1 x 4 x 640 uV), in order, after reset_state().
`Ensemble` computes the parameter-free features and the frozen FM embedding once per packet, shares them
across all members, averages member outputs and applies a monotone output calibration.
"""
import numpy as np
import torch

from .fm import GraphedFM
from .model import RF


class Calibration:
    """Monotone piecewise-linear map applied to the averaged countdown (identity when knots is None)."""

    def __init__(self, knots=None, values=None):
        if knots is not None:
            k, v = np.asarray(knots, float), np.asarray(values, float)
            if len(k) != len(v) or np.any(np.diff(k) <= 0) or np.any(np.diff(v) < 0):
                raise ValueError('calibration must be a monotone map on increasing knots')
        self.knots, self.values = knots, values

    @classmethod
    def blend(cls, knots, values, alpha):
        """Partial-strength map: knots -> knots + alpha * (values - knots)."""
        k = np.asarray(knots, float); v = np.asarray(values, float)
        return cls(k.tolist(), (k + alpha * (v - k)).tolist())

    def __call__(self, p):
        if self.knots is None:
            return p
        k = torch.as_tensor(self.knots, dtype=p.dtype, device=p.device)
        v = torch.as_tensor(self.values, dtype=p.dtype, device=p.device)
        i = torch.searchsorted(k, p.clamp(k[0], k[-1]).contiguous(), right=True).clamp(1, len(k) - 1)
        t = (p.clamp(k[0], k[-1]) - k[i - 1]) / (k[i] - k[i - 1])
        return v[i - 1] + t * (v[i] - v[i - 1])


class Ensemble:
    """Exact streaming mean of Sentinel members sharing one featurizer pass and one FM call per packet."""

    WIN = RF + 6                                  # TCN window >= receptive field + the 5 new seconds

    def __init__(self, nets, fm=None, calibration=None):
        self.nets = [n.eval() for n in nets]
        self.fm = GraphedFM(fm) if fm is not None else None
        if any(n.use_fm for n in self.nets) and fm is None:
            raise ValueError('FM members need the frozen encoder')
        self.calibration = calibration or Calibration()
        self.reset_state()

    def reset_state(self):
        self.prev = None; self.base = None
        self.u = [None] * len(self.nets); self.p = [None] * len(self.nets); self.hid = [None] * len(self.nets)

    @torch.no_grad()
    def predict_members(self, x):
        f = self.nets[0]
        x = x.reshape(1, 4, 640).float()
        fr = f.feat.frames(x, self.prev)
        self.prev = x[..., -128:].clone()
        b = f.feat.base(fr)
        # full history is kept: the trained cumulative-sum statistics are reproduced exactly
        self.base = b if self.base is None else {k: torch.cat((self.base[k], b[k]), 2) for k in b}
        chan, q, wave, pair = f.feat.derive(self.base)
        chan, q, wave, pair = chan[:, :, -5:], q[:, :, -5:], wave[:, :, -5:], pair[:, :, -5:]
        emb = self.fm(x).view(1, 1, 4, 512) if self.fm is not None else None
        outs = []
        for i, net in enumerate(self.nets):
            u, rel = net.seconds(chan, q, wave, pair)
            fmtok = net.fm_from_embed(emb, rel) if net.use_fm else None
            self.u[i] = u if self.u[i] is None else torch.cat((self.u[i], u), 1)[:, -self.WIN:]
            z = net.run_tcn(self.u[i])[:, -5:]
            pk = net.packets(z, fmtok)
            self.p[i] = pk if self.p[i] is None else torch.cat((self.p[i], pk), 1)
            s = net.slow_inputs(self.p[i])[:, -1:]
            if net.temporal == 'gru':
                h, self.hid[i] = net.gru(s, self.hid[i])
            else:
                cov = net.packet_coverage(rel, b)
                if self.hid[i] is None:
                    self.hid[i] = [None, None]
                h, self.hid[i][0] = net.memory[0].step(s, cov, self.hid[i][0])
                h, _ = net.condition_delay(h)
                h, self.hid[i][1] = net.memory[1].step(h, cov, self.hid[i][1])
            outs.append(net.heads(h)[0].reshape(1))
        return torch.stack(outs)                 # (members, 1)

    @torch.no_grad()
    def predict(self, x):
        return self.calibration(self.predict_members(x).mean(0))
