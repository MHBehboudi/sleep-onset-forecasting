"""Export a trained ensemble as a self-contained, single-file streaming predictor.

    python scripts/export.py --ensemble configs/ensemble_final.yaml --fm-dir weights/manas2 --out dist/sentinel

Writes dist/sentinel/{predictor.py, members.pt, fm_manas2.pt, calib.json} and dist/sentinel.zip.
predictor.py has no dependency on this repository (only torch and numpy): the source of
sentinel.features / fm / model / streaming is inlined, so the folder can be dropped onto a device or an
evaluation server as is.

    from predictor import load_model
    model = load_model('dist/sentinel', device='cuda')
    model.reset_state()                       # once per recording
    seconds = model.predict(packet)           # packet: (1, 4, 640) uV, [TP9, AF7, AF8, TP10] at 128 Hz
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sentinel.fm import load_manas2, save_compact        # noqa: E402

MODULES = ['features', 'fm', 'model', 'streaming']
LOADER = '''

# ======================= loader =======================
class Predictor:
    """Streaming wrapper: (B,4,640) packets on any device -> (B,) countdown in seconds."""

    def __init__(self, ens, device):
        self.ens, self.device = ens, device

    def reset_state(self):
        self.ens.reset_state()

    @torch.no_grad()
    def predict(self, x):
        x = torch.as_tensor(x).to(self.device, dtype=torch.float32)
        return torch.cat([self.ens.predict(x[b:b + 1]).reshape(1) for b in range(x.shape[0])])


def load_model(directory, device='cpu'):
    """Load the exported ensemble from `directory` onto `device`."""
    device = torch.device(device)
    root = Path(directory)
    pack = torch.load(root / 'members.pt', map_location='cpu', weights_only=True)
    nets = []
    for spec, state in zip(pack['specs'], pack['states']):
        net = Sentinel(temporal=spec['temporal'], fm=spec['fm'])
        net.load_state_dict(state, strict=True)
        nets.append(net.to(device).eval())
    fm = load_compact(root / 'fm_manas2.pt', device) if any(s['fm'] for s in pack['specs']) else None
    cal = json.loads((root / 'calib.json').read_text())
    model = Predictor(Ensemble(nets, fm=fm, calibration=Calibration(cal.get('knots'), cal.get('values'))), device)
    model.reset_state()
    return model
'''


def inline_source():
    parts = ['"""SENTINEL sleep-onset forecaster, single-file export. Uses only the EEG received so far in the\n'
             'current recording; reset_state() clears everything; elapsed time counts from the first packet."""\n'
             'import json\nfrom pathlib import Path\n']
    for m in MODULES:
        src = (ROOT / 'src' / 'sentinel' / f'{m}.py').read_text()
        src = re.sub(r'^from \.\w+ import [^\n]+\n', '', src, flags=re.M)     # intra-package imports
        parts.append(f'\n# ======================= sentinel/{m}.py =======================\n{src}')
    return ''.join(parts) + LOADER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble', required=True)
    ap.add_argument('--fm-dir', default='weights/manas2')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    E = yaml.safe_load(open(a.ensemble))
    a.out.mkdir(parents=True, exist_ok=True)
    specs, states = [], []
    for run in E['members']:
        info = json.load(open(Path(run) / 'member.json'))
        specs.append(dict(temporal=info['temporal'], fm=info['fm']))
        states.append(torch.load(Path(run) / 'final_ema.pt', map_location='cpu', weights_only=True))
    torch.save(dict(specs=specs, states=states), a.out / 'members.pt')
    if any(s['fm'] for s in specs):
        save_compact(load_manas2(a.fm_dir), a.out / 'fm_manas2.pt')
        lic = Path(a.fm_dir) / 'LICENSE'
        if lic.exists():
            shutil.copy(lic, a.out / 'LICENSE_MANAS2_Apache-2.0.txt')
    json.dump(E.get('calibration') or {}, open(a.out / 'calib.json', 'w'))
    (a.out / 'predictor.py').write_text(inline_source())
    shutil.make_archive(str(a.out), 'zip', a.out)
    print('wrote', a.out.with_suffix('.zip'))


if __name__ == '__main__':
    main()
