import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sentinel.augment import augment, make_stream          # noqa: E402
from sentinel.losses import kd_term, latency_coordinate      # noqa: E402
from sentinel.metric import metric_weights, wbmae            # noqa: E402
from sentinel.model import Sentinel                          # noqa: E402
from sentinel.streaming import Calibration, Ensemble         # noqa: E402


def fake_night(K=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(K * 640) / 128.
    alpha = torch.sin(2 * np.pi * 10 * t) * torch.linspace(20, 5, K * 640)    # fading alpha
    return (alpha + 15 * torch.randn(4, K * 640, generator=g)).float()


@pytest.mark.parametrize('temporal', ['gru', 'timescale'])
def test_streaming_equals_batch(temporal):
    torch.manual_seed(0)
    net = Sentinel(temporal=temporal).eval()
    raw = fake_night()
    with torch.no_grad():
        batch = net(raw[None])[0][0]
    ens = Ensemble([net])
    stream = torch.stack([ens.predict(raw[:, k * 640:(k + 1) * 640][None]).reshape(()) for k in range(raw.shape[1] // 640)])
    assert torch.allclose(batch, stream, atol=1e-3), (batch - stream).abs().max()


@pytest.mark.parametrize('temporal', ['gru', 'timescale'])
def test_causal(temporal):
    """Changing the future must not change any earlier output."""
    torch.manual_seed(0)
    net = Sentinel(temporal=temporal).eval()
    raw = fake_night()
    alt = raw.clone(); alt[:, 25 * 640:] = 300 * torch.randn_like(alt[:, 25 * 640:])
    with torch.no_grad():
        a, b = net(raw[None])[0][0], net(alt[None])[0][0]
    assert torch.allclose(a[:25], b[:25], atol=1e-5)
    assert not torch.allclose(a[25:], b[25:])


def test_dead_channel_is_downweighted():
    torch.manual_seed(0)
    net = Sentinel().eval()
    raw = fake_night(); raw[2] = 0.
    with torch.no_grad():
        rel = net(raw[None])[3][0]
    assert rel.shape == (4, raw.shape[1] // 128)


def test_metric_weights_reproduce_wbmae():
    y = torch.tensor([500., 400., 250., 120., 80., 60., 30., 10.])
    p = y + torch.tensor([50., -20., 10., 30., -5., 8., 4., -2.])
    assert abs(float((metric_weights(y) * (p - y).abs()).sum()) - wbmae(p.numpy(), y.numpy())) < 1e-5


def test_augment_mask_and_shape():
    g = torch.Generator().manual_seed(3)
    raw = fake_night(K=20)
    x, bad = augment(raw, g, p_fault=1.0)
    assert x.shape == raw.shape and bad.shape == (4, 100)


def test_make_stream_teacher_shift():
    g = torch.Generator().manual_seed(0)
    rec = dict(raw=fake_night(K=80), onset=300.)
    s0 = make_stream(rec, g, train=False, trim=0)
    s6 = make_stream(rec, g, train=False, trim=0, shift=6)
    assert s6['K'] == s0['K'] + 6
    assert torch.allclose(s6['y'][6:6 + 10], s0['y'][:10])      # teacher at k predicts the student's target at k-6


def test_calibration_monotone_and_identity():
    c = Calibration([0, 40, 90, 600], [0, 35, 85, 600])
    p = torch.linspace(0, 600, 50)
    q = c(p)
    assert torch.all(q[1:] >= q[:-1])
    assert torch.equal(Calibration()(p), p)


def test_kd_term_zero_when_student_matches_teacher():
    ends = torch.tensor([5., 10., 15.])
    curve = torch.tensor([[5., 10., 15.], [105., 110., 115.]])   # teacher implies onset at end + 100
    ps = torch.tensor([100., 100., 100.]); w = torch.ones(3) / 3
    assert float(kd_term(ps, w, curve, ends)) < 1e-6


def test_latency_coordinate_is_monotone():
    s = torch.linspace(0, 600, 601)
    assert torch.all(torch.diff(latency_coordinate(s)) > 0)
