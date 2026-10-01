"""Fixes #1 and #2: cached weight fake-quant and skipping the identity spike round-trip."""
import pytest
import torch
import torch.nn.functional as F

from conftest import tiny_model
import W8ASpike.quant_linear as ql

pytestmark = pytest.mark.skipif(not ql.spike_is_available, reason="Int2Spike (needs matplotlib) not importable")


def _activations(seed, dtype, shape=(3, 5, 512)):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g) * torch.rand(shape[:-1] + (1,), generator=g) * 4
    # heavy outliers push T (bits per spike train) up to ~10
    idx = torch.randint(0, x.numel(), (x.numel() // 100,), generator=g)
    x.view(-1)[idx] *= 50
    return x.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seed", range(20))
def test_spike_roundtrip_is_identity(dtype, seed):
    x = _activations(seed, dtype)
    ref_int, ref_vth = ql.dynamic_spikes(x, 3.0, spike_roundtrip=True)
    new_int, new_vth = ql.dynamic_spikes(x, 3.0, spike_roundtrip=False)
    assert ref_int.dtype == new_int.dtype
    assert torch.equal(ref_int, new_int) and torch.equal(ref_vth, new_vth)


def test_spike_roundtrip_identity_edge_cases():
    # one dominant activation per row hits the |x / vth| <= k * D upper bound; tiny rows hit the vth clamp
    x = torch.zeros(4, 512)
    x[0, 7] = 1e4
    x[1] = 1e-9
    x[2, :3] = torch.tensor([-1e6, 3.0, -2.0])
    x[3] = -torch.rand(512)
    ref, _ = ql.dynamic_spikes(x, 3.0, spike_roundtrip=True)
    new, _ = ql.dynamic_spikes(x, 3.0, spike_roundtrip=False)
    assert torch.equal(ref, new)


def test_fast_path_has_no_host_syncs(monkeypatch):
    layer = ql.QuantLinear(512, 256, bias=True).eval()
    x = torch.randn(2, 3, 512)
    layer(x)  # first call folds the weights

    def forbidden(*a, **k):
        raise AssertionError("host sync in QuantLinear fast path")

    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch, "allclose", forbidden)
    layer(x)


def _reference_quant_linear(layer, x, weight):
    """Original QuantLinear.forward: spike round-trip on, weight fake-quantized on the fly."""
    spikes_int, vth = ql.dynamic_spikes(x, layer.k, spike_roundtrip=True)
    xq = (spikes_int * vth).to(x.dtype)
    return F.linear(xq, layer.weight_quantizer(weight), layer.bias)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_quant_linear_matches_original(dtype):
    torch.manual_seed(0)
    layer = ql.QuantLinear(512, 256, bias=True).to(dtype).eval()
    w = layer.weight.detach().float().reshape(256, -1, 128)
    layer.weight_quantizer.scales.copy_(w.abs().amax(-1, keepdim=True) / 127)
    w_orig = layer.weight.detach().clone()

    calls = 0
    real_fwd = layer.weight_quantizer.forward

    def counting(weight):
        nonlocal calls
        calls += 1
        return real_fwd(weight)

    layer.weight_quantizer.forward = counting
    for step in range(3):
        x = _activations(step, dtype, (2, 4, 512))
        assert torch.equal(layer(x), _reference_quant_linear(layer, x, w_orig))
    assert calls == 1 + 3, "weights must be fake-quantized once (the +3 are the reference calls)"

    # new weights (e.g. load_state_dict) invalidate the cache and are re-folded
    new_w = torch.randn_like(w_orig)
    layer.load_state_dict({**layer.state_dict(), "weight": new_w})
    x = _activations(99, dtype, (2, 4, 512))
    assert torch.equal(layer(x), _reference_quant_linear(layer, x, new_w))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_refold_is_idempotent(dtype):
    """.to() swaps the storage and triggers a re-fold of already folded weights; that must be a no-op."""
    torch.manual_seed(1)
    q = ql.Quantizer(512, 256, 128).eval()
    w = (torch.randn(256, 512) * 0.05).to(dtype)
    q.scales.copy_(w.float().reshape(256, -1, 128).abs().amax(-1, keepdim=True) / 127)
    once = q(w)
    assert torch.equal(q(once), once)


def test_w8aspike_model_logits_unchanged_by_spike_skip(monkeypatch):
    model = tiny_model("W8ASpike")
    ids = torch.randint(3, 97, (2, 9))
    with torch.no_grad():
        fast = model(ids).logits
        monkeypatch.setattr(ql, "SPIKE_ROUNDTRIP", True)
        slow = model(ids).logits
    assert torch.equal(fast, slow)
