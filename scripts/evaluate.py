"""Score an ensemble on the development panel under test-like conditions, streaming packet by packet.

    python scripts/evaluate.py --ensemble configs/ensemble_final.yaml --conditions clean0 clean300 mix short

This uses exactly the inference path of the deployed model (sentinel.streaming.Ensemble), so a number
reported here is the number the packaged model produces. Each (night, condition) pair has its own
deterministic RNG, so every candidate sees the same cuts and faults.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sentinel.augment import make_stream                     # noqa: E402
from sentinel.data import load_panel, read_recording          # noqa: E402
from sentinel.fm import load_manas2                           # noqa: E402
from sentinel.metric import bin_mae, macro, wbmae             # noqa: E402
from sentinel.model import Sentinel                           # noqa: E402
from sentinel.streaming import Calibration, Ensemble          # noqa: E402


def seed_of(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def view(r, cond):
    g = torch.Generator().manual_seed(seed_of(r['id'] + cond))
    if cond == 'clean0':
        return make_stream(r, g, train=False, trim=0)
    if cond == 'clean300':
        return make_stream(r, g, train=False, trim=300) if r['onset'] - 300 >= 60 else None
    lo, hi_cap = (60, 240) if cond == 'short' else (120, 600)
    hi = int(min(hi_cap, r['onset'] - 60))
    trim = int(torch.randint(min(lo, hi), hi + 1, (1,), generator=g)) if hi > 0 else 0
    return make_stream(r, g, train=False, trim=trim, p_fault=0.35)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble', required=True)
    ap.add_argument('--data', default='data/nm000287')
    ap.add_argument('--panel', default='configs/dev_panel.json')
    ap.add_argument('--fm-dir', default='weights/manas2')
    ap.add_argument('--conditions', nargs='+', default=['clean300', 'mix'])
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    E = yaml.safe_load(open(a.ensemble))
    nets = []
    for run in E['members']:
        info = json.load(open(Path(run) / 'member.json'))
        n = Sentinel(temporal=info['temporal'], fm=info['fm'])
        n.load_state_dict(torch.load(Path(run) / 'final_ema.pt', map_location='cpu', weights_only=True))
        nets.append(n.to(dev))
    fm = load_manas2(a.fm_dir, dev) if any(n.use_fm for n in nets) else None
    cal = E.get('calibration') or {}
    ens = Ensemble(nets, fm=fm, calibration=Calibration(cal.get('knots'), cal.get('values')))
    panel = load_panel(a.panel)
    nights = [(i, 'seen') for i in panel['dev_seen']] + [(i, 'unseen') for i in panel['dev_unseen']]
    if a.limit:
        nights = nights[:a.limit]
    out = {}
    for cond in a.conditions:
        scores, groups, bins = [], [], []
        for rid, grp in nights:
            r = read_recording(a.data, rid, dev)
            s = view(r, cond)
            if s is None:
                continue
            ens.reset_state()
            p = np.array([float(ens.predict(s['raw'][:, k * 640:(k + 1) * 640][None])) for k in range(s['K'])])
            y = s['y'].cpu().numpy()
            scores.append(wbmae(p, y)); groups.append(grp); bins.append(bin_mae(p, y))
        m, g = macro(scores, groups)
        out[cond] = dict(macro=round(m, 3), **{k: round(v, 3) for k, v in g.items()}, nights=len(scores),
                         bin_mae=np.nanmean(bins, 0).round(2).tolist())
        print(cond, json.dumps(out[cond]), flush=True)
    return out


if __name__ == '__main__':
    main()
