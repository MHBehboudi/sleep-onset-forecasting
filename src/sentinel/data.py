"""Readers for the Muse sleep-onset dataset (nm000287, EEG-BIDS / BrainVision) and the external PSG
episodes used for pretraining.

The dataset is CC BY-NC-SA 4.0 and is not redistributed here: https://github.com/nemarDatasets/nm000287

Facts verified on all 540 recordings before any modelling:
  * 4 channels [TP9, AF7, AF8, TP10], 128 Hz, float32 with a header scale of 0.1 uV per unit;
  * exactly one `n2_onset` event; every recording ends exactly 300 s after it. The recording length
    therefore leaks the label; it is never a model input. The post-onset 300 s is used only as a
    training target for the auxiliary N2-state head;
  * onsets range from about 1 to 25 minutes after the start of the recording.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def read_recording(root, rid, device='cpu'):
    """rid 'sub-XXX_ses-YYY' -> dict(id, person, raw (4,S) uV incl. post-onset tail, onset s, n_pre packets)."""
    root = Path(root)
    s, ses = rid.split('_')
    eeg = root / s / ses / 'eeg'
    raw = np.fromfile(eeg / f'{s}_{ses}_task-sleeponset_eeg.eeg', np.float32).reshape(-1, 4).T * 0.1
    rows = [ln.split('\t') for ln in (eeg / f'{s}_{ses}_task-sleeponset_events.tsv').read_text(encoding='utf-8-sig').splitlines()]
    col = rows[0].index('onset'); kind = rows[0].index('trial_type')
    onset = float(next(r[col] for r in rows[1:] if r[kind] == 'n2_onset'))
    n_pre = int(onset // 5)
    K = raw.shape[1] // 640
    return dict(id=rid, person=s, onset=onset, n_pre=n_pre,
                raw=torch.from_numpy(np.ascontiguousarray(raw[:, :K * 640])).to(device))


def load_panel(path):
    return json.loads(Path(path).read_text())


def fold_of(person, nfolds=2):
    """Deterministic participant fold, used to cross-fit lookahead teachers."""
    return int(hashlib.sha256(('fold:' + person).encode()).hexdigest()[:8], 16) % nfolds


def load_source_episodes(path, device='cpu'):
    """External PSG sleep-onset episodes mapped into Muse slots (built by scripts/build_source_episodes.py).

    An episode is a contiguous wake/N1 block of >= 5 min containing >= 3 min of wake, followed by an N2 epoch,
    plus up to 300 s after it. Montage mapping (Muse records against Fpz): a frontal-to-mastoid derivation is
    approximately the negated mastoid-to-frontal Muse channel on the opposite side, so
      PhysioNet 2018: TP9 = -(F4-M1), TP10 = -(F3-M2), AF7/AF8 missing;
      BOAS          : AF7, AF8 = headband HB_1, HB_2; TP9 = -F4, TP10 = -F3.
    Missing slots are zeros and are marked bad so the reliability head is told.
    """
    z = np.load(path)
    raw, lens, meta = z['raw'], z['lens'], json.loads(str(z['meta']))
    out, pos = [], 0
    for L, m in zip(lens, meta):
        x = np.nan_to_num(np.clip(raw[:, pos:pos + L].astype(np.float32), -5000, 5000)); pos += L
        K = x.shape[1] // 640
        out.append(dict(id=f"{m['src']}:{m['rec']}", person=f"{m['src']}:{m['rec']}", onset=m['onset'],
                        n_pre=int(m['onset'] // 5), raw=torch.from_numpy(x[:, :K * 640]).to(device),
                        missing=torch.tensor(m['missing'], dtype=torch.bool)))
    return out
