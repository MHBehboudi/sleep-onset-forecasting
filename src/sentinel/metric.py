"""Recording-level weighted binned MAE (W-bMAE), the evaluation metric.

Within one recording, windows are binned by their TRUE remaining time to first N2:
[0,40) s, [40,90) s, [90,300) s and [300,600] s, with weights 10, 5, 3 and 1. The recording score is
sum(w_b * MAE_b) / sum(w_b) over non-empty bins. The blind test averages recording scores within
seen and unseen participants and reports the mean of the two group means.

Consequence used throughout the project: one window in the last 40 s carries the weight of about 120
windows in the far bin on a typical night (more on long nights). Near-onset accuracy dominates, and a
model that hedges far predictions toward the middle is behaving optimally, not failing.
"""
import numpy as np
import torch

EDGES = (0., 40., 90., 300., 600.0001)
BW = (10., 5., 3., 1.)


def metric_weights(y, bw=BW):
    """y (K,) targets of one recording's scored packets -> per-packet weights summing to 1.

    With these weights, sum(w * |p - y|) equals the recording's W-bMAE exactly."""
    w = torch.zeros_like(y); tot = 0.
    for i in range(4):
        m = (y >= EDGES[i]) & (y < EDGES[i + 1])
        n = int(m.sum())
        if n:
            w[m] = bw[i] / n; tot += bw[i]
    return w / tot


def wbmae(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float); num = den = 0.
    for i in range(4):
        m = (y >= EDGES[i]) & (y < EDGES[i + 1])
        if m.any():
            num += BW[i] * np.abs(p[m] - y[m]).mean(); den += BW[i]
    return num / den


def bin_mae(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return [float(np.abs(p[m] - y[m]).mean()) if m.any() else np.nan
            for m in ((y >= EDGES[i]) & (y < EDGES[i + 1]) for i in range(4))]


def macro(scores, groups):
    """Blind-test aggregation: mean of the seen-group mean and the unseen-group mean."""
    g = {k: float(np.mean([s for s, gg in zip(scores, groups) if gg == k])) for k in set(groups)}
    return float(np.mean(list(g.values()))), g
