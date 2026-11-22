"""Train one SENTINEL member from a recipe in configs/.

    python scripts/train.py --config configs/student_kd.yaml --seed 42 --out runs/student-s42
    python scripts/train.py --config configs/teacher.yaml --fold 0 --out runs/teacher-f0
    python scripts/train.py --config configs/teacher.yaml --fold 0 --init runs/teacher-f0/final_ema.pt \
        --dump-teacher cache/kd/teacher-f0.npz            # implied-onset curves for the held-out fold

Protocol (kept fixed for every comparison in the project): final epoch, EMA weights, no checkpoint
selection on the development panel. Selecting the best epoch had repeatedly produced gains that did not
replicate.
"""
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sentinel import losses as L                                     # noqa: E402
from sentinel.augment import make_stream                              # noqa: E402
from sentinel.data import fold_of, load_panel, load_source_episodes, read_recording   # noqa: E402
from sentinel.fm import embed_packets, load_manas2                    # noqa: E402
from sentinel.metric import bin_mae, macro, wbmae                     # noqa: E402
from sentinel.model import Sentinel                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--data', default='data/nm000287', help='nm000287 BIDS root')
ap.add_argument('--panel', default='configs/dev_panel.json')
ap.add_argument('--fm-dir', default='weights/manas2', help='folder with model.safetensors + positions.json')
ap.add_argument('--init', default=None, help='override the recipe init checkpoint')
ap.add_argument('--fold', type=int, default=None, help='teacher: train on participants whose fold != this')
ap.add_argument('--dump-teacher', default='', help='write held-out-fold implied-onset curves and exit')
ap.add_argument('--smoke', action='store_true', help='tiny CPU run: 6 nights, 1 epoch')
a = ap.parse_args()
R = yaml.safe_load(open(a.config))
if a.init is not None:
    R['init'] = a.init
if a.fold is not None:
    R['fold_train'] = a.fold
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(a.seed); np.random.seed(a.seed)
a.out.mkdir(parents=True, exist_ok=True)

# ---------------- data ----------------
panel = load_panel(a.panel)
ids = panel['fit'] + (panel['dev_seen'] + panel['dev_unseen'] + panel['public40'] if R.get('nights') == 'all' else [])
if R.get('fold_train', -1) >= 0:
    held_ids = [i for i in ids if fold_of(i.split('_')[0]) == R['fold_train']]
    ids = [i for i in ids if fold_of(i.split('_')[0]) != R['fold_train']]
dev_ids = [(i, 'seen') for i in panel['dev_seen']] + [(i, 'unseen') for i in panel['dev_unseen']] if R.get('nights') != 'all' else []
if a.smoke:
    ids, dev_ids, R['epochs'] = ids[:6], dev_ids[:2] + dev_ids[-2:], 1
    held_ids = held_ids[:4] if R.get('fold_train', -1) >= 0 else []
fit = [read_recording(a.data, i, dev) for i in ids]
dev_set = [dict(read_recording(a.data, i, dev), group=g) for i, g in dev_ids]
items = load_source_episodes(R['source'], dev) if R.get('source') else fit
print(f'{len(items)} training items, {len(dev_set)} development nights', flush=True)

# ---------------- model ----------------
net = Sentinel(d=128, slow=192, p=R.get('dropout', 0.1), temporal=R['temporal'], fm=R.get('fm', False)).to(dev)
with torch.no_grad():                         # standardizer on clean pre-onset fit nights
    acc = {'chan': [], 'q': [], 'pair': []}
    for r in fit[:48]:
        chan, q, _, pair = net.feat.derive(net.feat.base(net.feat.frames(r['raw'][:, :r['n_pre'] * 640][None])))
        acc['chan'].append(chan.reshape(-1, net.nc)); acc['q'].append(q.reshape(-1, net.nq)); acc['pair'].append(pair.reshape(-1, 12))
    for k, v in acc.items():
        net.std.fit(k, torch.cat(v))
if R.get('init'):                             # FM members start from the no-FM pretrained network (zero-init FM path)
    missing = net.load_state_dict(torch.load(R['init'], map_location=dev, weights_only=True), strict=False)
    print('init from', R['init'], 'new modules:', sorted({k.split('.')[0] for k in missing.missing_keys}), flush=True)
fm = load_manas2(a.fm_dir, dev) if R.get('fm') else None


def forward(model, streams):
    K = max(s['K'] for s in streams)
    raw = torch.stack([F.pad(s['raw'], (0, (K - s['K']) * 640)) for s in streams])
    emb = None
    if model.use_fm:
        B = raw.shape[0]
        emb = embed_packets(fm, raw.reshape(B, 4, K, 640).permute(0, 2, 1, 3).reshape(B * K, 4, 640)).view(B, K, 4, 512)
    return model(raw, emb)


shift = R.get('shift', 0)
if a.dump_teacher:                            # cross-fitting: curves only for nights the teacher never saw
    net.eval(); outd = {}
    with torch.no_grad():
        for rid in held_ids:
            r = read_recording(a.data, rid, dev)
            K = min(r['n_pre'] + shift + 1, r['raw'].shape[1] // 640)
            pt = forward(net, [dict(raw=r['raw'][:, :K * 640], K=K)])[0][0].cpu().numpy()
            te = 5. * np.arange(1, K + 1) - 5. * shift; ok = (te > 0) & (te < r['onset'])
            outd[rid] = np.stack((te[ok], te[ok] + pt[ok]))
    np.savez(a.dump_teacher, ids=np.array(list(outd)), **{f'c{i}': v for i, v in enumerate(outd.values())})
    print('dumped', len(outd), 'teacher curves to', a.dump_teacher); raise SystemExit

teachers = {}
for f in R.get('kd', []):
    z = np.load(f); teachers.update({str(k): torch.as_tensor(z[f'c{i}'], dtype=torch.float32) for i, k in enumerate(z['ids'])})
people = {p: i for i, p in enumerate(sorted({r['person'] for r in items}))}

opt = torch.optim.AdamW(net.parameters(), lr=R.get('lr', 1e-3), weight_decay=R.get('wd', 0.05))
steps_per = math.ceil(len(items) / R.get('batch', 16)); total = steps_per * R['epochs']; warm = steps_per
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1., (s + 1) / warm) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1., s / total)))))
ema = copy.deepcopy(net).eval()
for p_ in ema.parameters():
    p_.requires_grad_(False)
aug = dict(gain_p=R.get('gain_p', 0.8), gain_lo=R.get('gain_lo', 0.4), gain_hi=R.get('gain_hi', 2.5), chan_sd=R.get('chan_sd', 0.25))
g = torch.Generator().manual_seed(a.seed); prefix_rng = torch.Generator().manual_seed(a.seed + 1)
log = open(a.out / 'log.jsonl', 'a'); step = 0

for ep in range(1, R['epochs'] + 1):
    order = torch.randperm(len(items), generator=g).tolist(); t0 = time.time(); agg = {}
    for bi in range(steps_per):
        idx = order[bi * R.get('batch', 16):(bi + 1) * R.get('batch', 16)]
        streams = [make_stream(items[i], g, train=True, p_fault=R.get('p_fault', 0.4), trim_max=R.get('trim_max', 900), shift=shift, aug=aug) for i in idx]
        point, hz, st, rel = forward(net, streams)
        n = len(streams); tot = dict(point=0., hazard=0., state=0., rel=0., kd=0.)
        Kmax = point.shape[1]
        tgt = torch.zeros_like(point); wgt = torch.zeros_like(point)
        for j, s in enumerate(streams):
            Kj = s['K']
            c = L.core_terms((point[j, :Kj], hz[j, :Kj], st[j, :Kj], rel[j, :, :Kj * 5]), s, R.get('near_w', 1.0))
            if R.get('latency_point'):
                c['point'] = L.latency_point_loss(c['ps'], c['ys'], c['w'])
            for k in ('point', 'hazard', 'state', 'rel'):
                tot[k] = tot[k] + c[k]
            rid = items[idx[j]]['id']
            if R.get('kd_w', 0) > 0 and rid in teachers:
                tot['kd'] = tot['kd'] + L.kd_term(c['ps'], c['w'], teachers[rid], s['ends'][:Kj][s['scored']])
            tgt[j, :Kj][s['scored']] = c['ys']; wgt[j, :Kj][s['scored']] = c['w']
        loss = tot['point'] / n / 50. + 0.1 * tot['hazard'] / n + 0.2 * tot['state'] / n + 0.1 * tot['rel'] / n \
            + R.get('kd_w', 0) * tot['kd'] / n / 50.
        if R.get('delay_rank_w', 0) > 0:
            who = torch.tensor([people[items[i]['person']] for i in idx], device=dev)
            loss = loss + R['delay_rank_w'] * L.matched_delay_loss(net.delay_score, point, tgt, wgt, who)
        if R.get('prefix_w', 0) > 0:
            loss = loss + R['prefix_w'] * L.prefix_loss(net, streams, tgt, wgt, prefix_rng) / 50.
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        with torch.no_grad():
            d = min(0.995, (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(d).add_(pn.detach(), alpha=1 - d)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
        for k, v in tot.items():
            agg[k] = agg.get(k, 0.) + float(v) / n / steps_per
    row = dict(epoch=ep, sec=round(time.time() - t0, 1), lr=sched.get_last_lr()[0], **{k: round(v, 4) for k, v in agg.items()})
    print(json.dumps(row), flush=True); log.write(json.dumps(row) + '\n'); log.flush()

torch.save(ema.state_dict(), a.out / 'final_ema.pt')
json.dump(dict(config=R, seed=a.seed, temporal=R['temporal'], fm=bool(R.get('fm'))), open(a.out / 'member.json', 'w'), indent=1)


def seed_of(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


if dev_set:                                   # fixed-epoch development score under two test-like views
    res = {}
    with torch.no_grad():
        for cond, trim, pf in (('clean300', 300, None), ('mix', None, 0.35)):
            scores, groups, bins = [], [], []
            for r in dev_set:
                gg = torch.Generator().manual_seed(seed_of(r['id'] + cond))
                if cond == 'clean300' and r['onset'] - 300 < 60:
                    continue
                t = trim if trim is not None else int(torch.randint(min(120, int(min(600, r['onset'] - 60))), int(min(600, r['onset'] - 60)) + 1, (1,), generator=gg))
                s = make_stream(r, gg, train=False, trim=t, p_fault=pf)
                p = forward(ema, [s])[0][0].cpu().numpy(); y = s['y'].cpu().numpy()
                scores.append(wbmae(p, y)); groups.append(r['group']); bins.append(bin_mae(p, y))
            m, grp = macro(scores, groups)
            res[cond] = dict(macro=round(m, 3), **{k: round(v, 3) for k, v in grp.items()}, bin_mae=np.nanmean(bins, 0).round(2).tolist())
    print(json.dumps(res)); json.dump(res, open(a.out / 'final_dev.json', 'w'), indent=1)
