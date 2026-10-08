"""Fixes #1 and #2: cached weight fake-quant and skipping the identity spike round-trip."""
import copy
import warnings

import pytest
import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_spike_roundtrip_identity_edge_cases(dtype):
    # one dominant activation per row hits the |x / vth| <= k * D upper bound; tiny rows hit the
    # 1e-5 vth min clamp; 1e12-scale rows hit the 1e4 vth max clamp, the only route to spike
    # magnitudes >= 2**24 (beyond the fp32 integer range)
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(5, 512)
    x[0, 7] = 1e4
    x[1] = 1e-9
    x[2, :3] = torch.tensor([-1e6, 3.0, -2.0])
    x[3] = -torch.rand(512, generator=g)
    x[4] = torch.randn(512, generator=g) * 1e12
    x = x.to(dtype)
    ref, vth = ql.dynamic_spikes(x, 3.0, spike_roundtrip=True)
    new, _ = ql.dynamic_spikes(x, 3.0, spike_roundtrip=False)
    assert vth.min() == 1e-5 and vth.max() == 1e4, "both vth clamps must be exercised"
    assert new[4].abs().max() >= 2 ** 24
    assert torch.equal(ref, new)


def test_spike_roundtrip_domain_boundary():
    """The original path drops the MSB when max|x| is an exact power of two >= 2**49 (double rounding
    in math.log2); below that bound the skip and the round-trip agree, as the module comment states."""
    ok = torch.tensor([[2.0 ** 48, -(2.0 ** 48), 2.0 ** 48 - 2 ** 25, 5.0, -3.0]])
    assert torch.equal(ql.dynamic_spikes(ok * 1e4, 3.0, spike_roundtrip=True)[0],
                       ql.dynamic_spikes(ok * 1e4, 3.0, spike_roundtrip=False)[0])
    bad = torch.tensor([[2.0 ** 50, -(2.0 ** 50), 5.0, -3.0]]) * 1e4  # vth clamps to 1e4 -> spikes 2**50
    slow, _ = ql.dynamic_spikes(bad, 3.0, spike_roundtrip=True)
    fast, _ = ql.dynamic_spikes(bad, 3.0, spike_roundtrip=False)
    assert fast[0, 0] == 2.0 ** 50 and slow[0, 0] == 0  # the skip is the correct one here


SYNC_OPS = {"item", "__bool__", "__int__", "__float__", "__index__", "tolist", "numpy", "allclose",
            "equal", "nonzero", "is_nonzero", "__len__"}


class _OpRecorder(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.ops.append(getattr(func, "__name__", str(func)))
        return func(*args, **(kwargs or {}))


def _sync_ops(fn):
    with _OpRecorder() as rec:
        fn()
    return sorted(set(rec.ops) & SYNC_OPS)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fast_path_has_no_host_syncs(dtype):
    layer = ql.QuantLinear(512, 256, bias=True).to(dtype).eval()
    x = torch.randn(2, 3, 512).to(dtype)
    layer(x)  # first call folds the weights
    assert _sync_ops(lambda: layer(x)) == []
    # positive control: the detector sees the syncs of the explicit spike path
    layer.spike_roundtrip = True
    assert {"allclose", "__bool__", "item"} <= set(_sync_ops(lambda: layer(x)))


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


@pytest.mark.parametrize("qmax", [127, 255])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_refold_is_idempotent(dtype, qmax):
    """.to() / deepcopy swap the storage and trigger a re-fold of already folded weights with the
    same scales; that must be a no-op. Goes through QuantLinear so the scales buffer has the
    module dtype (bf16 scales for a bf16 model), with |W / scale| up to the documented 255."""
    torch.manual_seed(1)
    layer = ql.QuantLinear(512, 256, bias=True).to(dtype).eval()
    assert layer.weight_quantizer.scales.dtype == dtype
    w = layer.weight.detach().clone().float().reshape(256, -1, 128)
    layer.weight_quantizer.scales.copy_(w.abs().amax(-1, keepdim=True) / qmax)
    x = torch.randn(2, 3, 512).to(dtype)
    layer(x)
    folded = layer.weight.detach().clone()
    assert layer._wq_key is not None and not torch.equal(folded, w.reshape(256, 512).to(dtype))

    twin = copy.deepcopy(layer)
    assert twin._weight_key() != layer._wq_key
    twin(x)  # re-folds the already folded weight
    assert torch.equal(twin.weight, folded)
    assert torch.equal(twin(x), layer(x))


def test_w8aspike_model_logits_unchanged_by_spike_skip(monkeypatch):
    model = tiny_model("W8ASpike")
    ids = torch.randint(3, 97, (2, 9))
    with torch.no_grad():
        fast = model(ids).logits
        monkeypatch.setattr(ql, "SPIKE_ROUNDTRIP", True)
        slow = model(ids).logits
    assert torch.equal(fast, slow)


def _calibrated_layer(dtype=torch.float32, seed=0, **kw):
    torch.manual_seed(seed)
    layer = ql.QuantLinear(512, 256, bias=True, **kw).to(dtype).eval()
    w = layer.weight.detach().float().reshape(256, -1, 128)
    layer.weight_quantizer.scales.copy_(w.abs().amax(-1, keepdim=True) / 127)
    return layer


def _build_inference_layer(how):
    """Realistic ways a QuantLinear ends up with inference-tensor weight/scales."""
    if how == "built_inside":
        with torch.inference_mode():
            return _calibrated_layer()
    if how == "to_dtype_inside":
        layer = _calibrated_layer()
        with torch.inference_mode():
            return layer.to(torch.bfloat16)
    if how == "deepcopy_inside":
        layer = _calibrated_layer()
        with torch.inference_mode():
            return copy.deepcopy(layer)
    raise ValueError(how)


@pytest.mark.parametrize("forward_inside", [True, False])
@pytest.mark.parametrize("how", ["built_inside", "to_dtype_inside", "deepcopy_inside"])
def test_quant_linear_with_inference_tensors(how, forward_inside):
    """Weight/scales created under torch.inference_mode() have no version counter and cannot be
    written in place outside it; the layer must still work (uncached), inside and outside."""
    layer = _build_inference_layer(how)
    assert layer.weight.is_inference() and layer.weight_quantizer.scales.is_inference()
    w_src = layer.weight.detach().clone()
    x = torch.randn(2, 3, 512).to(layer.weight.dtype)
    ctx = torch.inference_mode() if forward_inside else torch.no_grad()
    with ctx:
        out1 = layer(x)
        out2 = layer(x)
        ref = _reference_quant_linear(layer, x, w_src)
    assert torch.equal(out1, out2) and torch.equal(out1, ref)
    assert layer._wq_key is None and torch.equal(layer.weight, w_src), "no fold on inference tensors"


def test_quant_linear_inference_tensor_tracks_inplace_writes():
    """Uncached means an in-place reload under inference_mode is honoured (no stale key)."""
    with torch.inference_mode():
        layer = _calibrated_layer()
        x = torch.randn(2, 3, 512)
        layer(x)
        new_w = torch.randn_like(layer.weight)
        layer.weight.copy_(new_w)
        assert torch.equal(layer(x), _reference_quant_linear(layer, x, new_w))


def test_quant_linear_warns_once_for_inference_tensors(monkeypatch):
    monkeypatch.setattr(ql, "_INFERENCE_TENSOR_WARNED", False)
    with torch.inference_mode():
        layer = _calibrated_layer()
        x = torch.randn(2, 3, 512)
        with pytest.warns(UserWarning, match="inference tensors"):
            layer(x)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # only once per process
            layer(x)


def test_w8aspike_model_built_and_run_under_inference_mode():
    """`with torch.inference_mode(): model = ...to(bf16); model(ids)` is a standard inference
    pattern; it must give the same logits as building outside and running under no_grad."""
    ids = torch.randint(3, 97, (2, 9))
    with torch.no_grad():
        ref = tiny_model("W8ASpike").to(torch.bfloat16)(ids).logits
    with torch.inference_mode():
        model = tiny_model("W8ASpike").to(torch.bfloat16)
        assert all(m.weight.is_inference() for m in model.modules() if isinstance(m, ql.QuantLinear))
        out = model(ids).logits
        gen = model.generate(ids, max_new_tokens=3, do_sample=False)
    assert torch.equal(out, ref) and gen.shape == (2, 12)


def test_scale_only_change_after_fold_raises():
    """CodeRabbit B: the in-place fold destroys the source weight, so re-quantizing with new scales
    would run on the already quantized grid (0.014 @0.01 -> 0.01, then @0.02 -> 0 instead of 0.02)."""
    layer = ql.QuantLinear(256, 8, bias=False).eval()
    x = torch.randn(2, 4, 256)
    with torch.no_grad():
        layer.weight.fill_(0.014)
        layer.weight_quantizer.scales.fill_(0.01)
    layer(x)
    assert torch.allclose(layer.weight, torch.full_like(layer.weight, 0.01))
    with torch.no_grad():
        layer.weight_quantizer.scales.fill_(0.02)
    with pytest.raises(ValueError, match="scales changed after the weight was folded"):
        layer(x)
    assert torch.allclose(layer.weight, torch.full_like(layer.weight, 0.01)), "weight left untouched"

    # a scales-only checkpoint (strict=False) after a forward is the same situation
    layer = _calibrated_layer()
    layer(torch.randn(2, 4, 512))
    new_scales = layer.weight_quantizer.scales * 2
    layer.load_state_dict({"weight_quantizer.scales": new_scales}, strict=False)
    with pytest.raises(ValueError, match="(?i)reload the source weights"):
        layer(torch.randn(2, 4, 512))


def test_scale_change_with_weight_reload_is_fine():
    """Reloading weight + scales together (full load_state_dict) re-folds from the new source."""
    layer = ql.QuantLinear(256, 8, bias=False).eval()
    x = torch.randn(2, 4, 256)
    with torch.no_grad():
        layer.weight.fill_(0.014)
        layer.weight_quantizer.scales.fill_(0.01)
    layer(x)
    layer.load_state_dict({"weight": torch.full_like(layer.weight, 0.014),
                           "weight_quantizer.scales": torch.full_like(layer.weight_quantizer.scales, 0.02)})
    layer(x)
    assert torch.allclose(layer.weight, torch.full_like(layer.weight, 0.02))

    # weight-only reload keeps the old scales and is fine too
    layer = _calibrated_layer()
    xx = torch.randn(2, 4, 512)
    layer(xx)
    new_w = torch.randn_like(layer.weight)
    layer.load_state_dict({"weight": new_w}, strict=False)
    assert torch.equal(layer(xx), _reference_quant_linear(layer, xx, new_w))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_to_dtype_after_fold_does_not_raise(dtype):
    """.to(dtype) recreates the fp32 scales buffer in the new dtype (same values) even when the
    weight tensor is kept; that is a legitimate (idempotent) re-fold, not a scale change."""
    layer = _calibrated_layer(dtype)
    x = torch.randn(2, 3, 512).to(dtype)
    layer(x)
    folded = layer.weight.detach().clone()
    other = torch.float64 if dtype == torch.float32 else torch.float32  # lossless round trip
    layer.to(other).to(dtype)
    layer(x)
    assert torch.equal(layer.weight, folded)
    # scales buffer recreated in bf16 while the bf16 weight stays in place
    layer = _calibrated_layer(torch.bfloat16)
    layer.weight_quantizer.scales = layer.weight_quantizer.scales.float()
    layer(x.to(torch.bfloat16))
    w_ptr = layer.weight.data_ptr()
    layer.to(torch.bfloat16)
    assert layer.weight.data_ptr() == w_ptr and layer.weight_quantizer.scales.dtype == torch.bfloat16
    layer(x.to(torch.bfloat16))


def test_keep_source_weights_allows_recalibration():
    """Opt-in: the source weight is never modified, the quantized copy is cached separately and
    recomputed when the scales (or the weight) change."""
    layer = _calibrated_layer(keep_source_weights=True)
    w_src = layer.weight.detach().clone()
    x = torch.randn(2, 3, 512)
    calls = 0
    real_fwd = layer.weight_quantizer.forward

    def counting(weight):
        nonlocal calls
        calls += 1
        return real_fwd(weight)

    layer.weight_quantizer.forward = counting
    assert torch.equal(layer(x), _reference_quant_linear(layer, x, w_src))
    layer(x)
    assert calls == 1 + 1 and torch.equal(layer.weight, w_src)

    with torch.no_grad():
        layer.weight_quantizer.scales.mul_(2)  # re-calibration after the first forward
    assert torch.equal(layer(x), _reference_quant_linear(layer, x, w_src))
    assert calls == 2 + 2 and torch.equal(layer.weight, w_src)  # one recompute (+1 reference call)

    twin = copy.deepcopy(layer).to(torch.bfloat16)  # storage move: cache recomputed from the source
    assert torch.equal(twin(x.bfloat16()), _reference_quant_linear(twin, x.bfloat16(), w_src.bfloat16()))
    assert torch.equal(twin.weight, w_src.bfloat16())


def test_keep_source_weights_after_fold_rejects_new_scales():
    """Switching keep_source_weights on after an in-place fold cannot recover the source weight, so a
    later scale change must raise rather than quantize the folded grid again."""
    layer = ql.QuantLinear(256, 8, bias=False).eval()
    x = torch.randn(2, 4, 256)
    with torch.no_grad():
        layer.weight.fill_(0.014)
        layer.weight_quantizer.scales.fill_(0.01)
    layer(x)  # in-place fold with scales 0.01
    layer.keep_source_weights = True
    layer(x)  # same scales: cached copy of the (already folded) weight, no error
    with torch.no_grad():
        layer.weight_quantizer.scales.fill_(0.02)
    with pytest.raises(ValueError, match="keep_source_weights was enabled after"):
        layer(x)
    assert torch.allclose(layer.weight, torch.full_like(layer.weight, 0.01)), "weight left untouched"
