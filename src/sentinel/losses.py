"""Training objectives.

Every member is trained with the same core objective, computed per recording so that each night counts
once (the metric is a mean over recordings, not windows):

    L = point/50 + 0.1 * hazard + 0.2 * post-onset state + 0.1 * reliability  [+ member-specific terms]

point      metric-weighted L1 on the countdown: sum_k w_k |p_k - y_k| equals the recording's W-bMAE.
           `near_w` multiplies the [0,40) s bin weight (near-weighted members use 3).
hazard     121-way cross-entropy over 5 s bins of the clipped target, label smoothing 0.1. Dense
           ordinal supervision for the encoder; not used at inference.
state      BCE "has N2 started" on the 300 s of post-onset EEG every training recording contains.
           Never available at inference; it teaches the encoder what N2 looks like.
reliability BCE of each channel's per-second reliability against the known corruption mask.

Member-specific terms:
kd         lookahead distillation: L1 toward a cross-fitted TEACHER that saw 30 s of the future
           (same metric weights). Worth about -0.4 s per model on the development panel.
latency    monotone re-coordinatisation g(s) = s + max(s-80,0)^2/400 of the point loss: more gradient
           for long delays while keeping the same weighted-median optimum.
delay rank pairwise BCE between packets of DIFFERENT people at the same received time, one 90-240 s and one
           300-600 s from onset, that the model currently confuses (|p_i - p_j| <= 45 s).
prefix     extra L1 on short histories (1-18 packets) cut at metric-weighted endpoints: experience with
           "just switched on, close to sleep", which natural nights almost never provide.
"""
import torch
from torch.nn import functional as F

from .metric import BW, metric_weights


def core_terms(out, stream, near_w=1.0):
    """out: (point, hazard, state, rel) for ONE stream (already sliced to its K packets)."""
    point, hz, st, rel = out
    y, scored, bad = stream['y'], stream['scored'], stream['bad']
    ys, ps = y[scored], point[scored]
    w = metric_weights(ys, (BW[0] * near_w, *BW[1:]))
    lp = (w * (ps - ys).abs()).sum()
    cls = (ys / 5).floor().clamp(max=120).long()
    lh = F.cross_entropy(hz[scored], cls, label_smoothing=0.1)
    ls = F.binary_cross_entropy_with_logits(st, (~scored).float())
    lr = F.binary_cross_entropy_with_logits(rel, 1. - bad)
    return dict(point=lp, hazard=lh, state=ls, rel=lr, w=w, ys=ys, ps=ps)


def kd_term(ps, w, teacher_curve, ends):
    """teacher_curve (2, T): teacher end times and implied ONSET times (end + predicted countdown).
    Interpolated at the student's packet ends, converted back to a countdown, compared with L1."""
    import numpy as np
    e = ends.float().cpu().numpy()
    onset_hat = np.interp(e, teacher_curve[0].numpy(), teacher_curve[1].numpy())
    target = (torch.from_numpy(onset_hat).to(ps.device) - ends.to(ps.device)).clamp(0, 600).float()
    return (w * (ps - target).abs()).sum()


def latency_coordinate(seconds):
    above = (seconds - 80.).clamp_min(0.)
    return seconds + above.square() / 400.


def latency_point_loss(ps, ys, w):
    return (w * (latency_coordinate(ps) - latency_coordinate(ys)).abs() / 1.15).sum()


def matched_delay_loss(logit, point, target, weight, participant):
    """Pairwise ranking on hard long-delay pairs. All inputs (B, K); participant (B,)."""
    medium = (target >= 90) & (target < 240) & (weight > 0)
    distant = (target >= 300) & (target < 600) & (weight > 0)
    different = participant[:, None] != participant[None, :]
    close = (point.detach()[None] - point.detach()[:, None]).abs() <= 45
    eligible = medium[:, None] & distant[None] & different[..., None] & close
    left = eligible.sum(dim=1, keepdim=True).clamp_min(1)
    right = eligible.sum(dim=0, keepdim=True).clamp_min(1)
    pair_w = eligible * ((weight[:, None] * weight[None]) / (left * right)).sqrt()
    pref = ((target[None] - target[:, None]) / 120).sigmoid()
    loss = F.binary_cross_entropy_with_logits(logit[None] - logit[:, None], pref, reduction='none')
    return (loss * pair_w).sum() / pair_w.sum().clamp_min(1e-8)


def prefix_loss(model, streams, targets, weights, rng):
    """Short received histories for the first half of the batch (fresh model state for each clip)."""
    count = (len(streams) + 1) // 2
    positions = torch.multinomial(weights[:count].detach().cpu(), 1, generator=rng).flatten()
    clips, lengths = [], []
    for index, position in enumerate(positions.tolist()):
        end = position + 1
        length = int(torch.randint(1, min(end, 18) + 1, (), generator=rng))
        lengths.append(length)
        clips.append(streams[index]['raw'][:, (end - length) * 640:end * 640])
    width = max(lengths) * 640
    batch = torch.stack([F.pad(c, (0, width - c.shape[-1])) for c in clips])
    pred = model(batch)[0]
    index = torch.arange(count, device=pred.device)
    last = torch.tensor(lengths, device=pred.device) - 1
    return (pred[index, last] - targets[index, positions.to(pred.device)]).abs().mean()
