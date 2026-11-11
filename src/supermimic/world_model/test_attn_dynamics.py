import torch
import pytest

from supermimic.world_model.attn_dynamics import (
    AttnDynCfg, HybridDynamics, TokenizedDelta, joint_features, scatter_delta,
)
from supermimic.world_model.dynamics import DynamicsMLP

SMALL = AttnDynCfg(d_model=32, n_heads=4)


def _inputs(n=8):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(n, 71, generator=g), torch.randn(n, 29, generator=g), torch.randn(n, 33, generator=g))


def test_gather_columns():
    feat = torch.arange(71.0).expand(2, 71)
    act = (1000 + torch.arange(29.0)).expand(2, 29)
    xi = (2000 + torch.arange(33.0)).expand(2, 33)
    out = joint_features(feat, act, xi)
    for j in range(29):
        assert out[0, j].tolist() == [13 + j, 42 + j, 1000 + j, 2000 + 1 + j]


def test_scatter_layout():
    r = torch.arange(12.0).expand(3, 12)
    j = torch.stack([100 + torch.arange(29.0), 200 + torch.arange(29.0)], dim=-1).expand(3, 29, 2)
    d = scatter_delta(r, j)
    assert d.shape == (3, 70)
    assert torch.equal(d[0, 0:3], r[0, 0:3])      # dv_h
    assert torch.equal(d[0, 3:6], r[0, 3:6])      # dw_b
    assert torch.equal(d[0, 6:35], 100 + torch.arange(29.0))   # dqd
    assert torch.equal(d[0, 35:38], r[0, 6:9])    # dp_h
    assert torch.equal(d[0, 38:41], r[0, 9:12])   # dth_b
    assert torch.equal(d[0, 41:70], 200 + torch.arange(29.0))  # dq


def test_zero_branch_equals_baseline():
    torch.manual_seed(0)
    h = HybridDynamics(hidden=(64, 64, 64), attn_cfg=SMALL)
    for p in h.mlp.net[-1].parameters():          # make the MLP output non-trivial
        torch.nn.init.normal_(p, std=0.1)
    x = _inputs()
    assert torch.equal(h(*x), DynamicsMLP.forward(h, *x))


def test_attn_params_receive_gradients():
    m = TokenizedDelta(SMALL)
    for head in (m.root_head, m.joint_head):
        torch.nn.init.normal_(head.weight, std=0.1)
    m(*_inputs()).pow(2).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_baseline_checkpoint_loads_into_hybrid():
    base = DynamicsMLP(hidden=(64, 64, 64))
    hyb = HybridDynamics(hidden=(64, 64, 64), attn_cfg=SMALL)
    missing, unexpected = hyb.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("attn.") for k in missing)


def test_temporal_guard():
    with pytest.raises(ValueError):
        TokenizedDelta(AttnDynCfg(depth=3, time_every=3))   # R3 setting is not supported yet


@pytest.mark.parametrize("lead", [(5,), (2, 3)])
def test_leading_dims(lead):
    m = TokenizedDelta(SMALL)
    f, a, x = torch.randn(*lead, 71), torch.randn(*lead, 29), torch.randn(*lead, 33)
    assert m(f, a, x).shape == (*lead, 70)
