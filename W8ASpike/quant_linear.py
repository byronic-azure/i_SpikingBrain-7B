import os
import warnings

import torch
import torch.nn as nn

try:
    from .Int2Spike.neuron import spike_fake_quant, SpikeCountBitwiseNode, spike_matmul
    spike_is_available = True
except Exception as e:
    print('need https://github.com/BICLab/Int2Spike repo to do fake int2spike, ', e)
    spike_is_available = False

# The bidirectional bitwise spike encode->decode in Int2Spike is an exact identity on
# integer-valued inputs with max|x| < 2**49: T = ceil(log2(max|x| + 1)) bits suffice, the
# decode sums powers of two, and every partial sum of the set bits of an fp32 integer is
# itself exactly representable in fp32. Outside that domain the original path is the one
# that is lossy or raises: for max|x| = 2**k with k >= 49, math.log2(2**k + 1) rounds to k
# in double, T comes out one bit short and the maximum-magnitude elements decode to 0;
# non-finite inputs decode to 0 (inf) or raise (nan); an all-non-negative tensor raises.
# (With the 1e4 cap on vth that needs |x| >= ~5.6e18.) Running the round-trip costs ~T
# extra fp32/int64 passes and 3+ host syncs per linear layer, so by default we skip it.
# Set W8ASPIKE_SPIKE_ROUNDTRIP=1 (or pass spike_roundtrip=True) to run the explicit spike
# path, e.g. to collect firing-rate or sparsity statistics.
SPIKE_ROUNDTRIP = os.environ.get("W8ASPIKE_SPIKE_ROUNDTRIP", "0") == "1"


def dynamic_spikes(x, k=3.0, spike_roundtrip=None):
    vth = x.abs().mean([-1], keepdim=True).float() / k
    vth = vth.clamp(min=1e-5, max=1e4)
    spikes_int = (x / vth).round()

    if spike_roundtrip is None:
        spike_roundtrip = SPIKE_ROUNDTRIP
    if spike_roundtrip and spike_is_available:
        spikes_int = spike_fake_quant(spikes_int, lif_quantizer=SpikeCountBitwiseNode(is_bidirectional=True))

    return spikes_int, vth

_INFERENCE_TENSOR_WARNED = False


class QuantLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, w_group_size=128, dynamic_sfr=3.0,
                 keep_source_weights: bool = False):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)

        self.k = dynamic_sfr
        self.w_group_size = w_group_size
        self.weight_quantizer = Quantizer(in_features, out_features, w_group_size)
        self.spike_roundtrip = None  # None -> follow the module-level SPIKE_ROUNDTRIP default
        # keep_source_weights=True keeps ``self.weight`` untouched and caches the fake-quantized
        # weight in a separate tensor (a second weight-sized copy). Use it for calibration
        # workflows that change ``weight_quantizer.scales`` after the layer has run. The default
        # folds in place (memory-flat) and then rejects scale-only changes, see quantized_weight().
        self.keep_source_weights = keep_source_weights
        self._wq_key = None
        self._wq = None            # cached fake-quantized weight when keep_source_weights=True
        self._fold_scales = None   # copy of the scales the in-place fold was computed with

    def _weight_key(self):
        w, s = self.weight, self.weight_quantizer.scales
        return (w.data_ptr(), w._version, w.device, w.dtype, s.data_ptr(), s._version)

    def _scales_changed_since_fold(self):
        """True when the scales differ in value from the ones the in-place fold used.

        A pure ``.to(dtype)`` / ``.to(device)`` creates a new scales buffer with the same values
        (up to the dtype cast), which must not count as a change: compare in the current dtype.
        """
        s = self.weight_quantizer.scales
        if self._fold_scales is None:
            return False  # no in-place fold happened (e.g. keep_source_weights was True so far)
        if self._fold_scales.shape != s.shape:
            return True
        return not torch.equal(self._fold_scales.to(device=s.device, dtype=s.dtype), s)

    def quantized_weight(self):
        """Return the fake-quantized weight, computing it once per weight/scales state.

        Default (``keep_source_weights=False``): the result is folded IN PLACE into
        ``self.weight`` so memory stays flat (no second 7B-sized copy). The fold is redone
        whenever the weight or scales storage changes (``load_state_dict``, ``.to()``, any
        in-place write), tracked by data pointer and version counter. Re-folding already
        folded weights with the same scales is idempotent (bf16/fp16/fp32 weights with fp32
        or same-dtype scales), so moves and reloads are exact.

        The fold destroys the source weight, so the scales can only be changed together with
        a reload of the source weights (e.g. a full ``load_state_dict`` with both ``weight``
        and ``weight_quantizer.scales``). A scale change while the weight is unchanged would
        silently re-quantize the already quantized grid, so it raises ``ValueError`` instead.
        Limitation: a scale change that is followed by a weight storage move (``.to()``)
        before the next forward looks like a reload and is not detected.

        ``keep_source_weights=True``: ``self.weight`` is never modified; the quantized weight
        is cached in ``self._wq`` and recomputed from the source whenever the weight or
        scales change. Costs one extra weight-sized tensor.

        Inference tensors (weight or scales created under ``torch.inference_mode()``, e.g.
        a model built or loaded inside it) have no version counter and cannot be written in
        place outside inference mode, so they bypass the cache: the weight is fake-quantized
        on every forward, like the original code. Build/load the model outside
        ``torch.inference_mode()`` (running it inside is fine) to get the cached fold.
        """
        global _INFERENCE_TENSOR_WARNED
        w, s = self.weight, self.weight_quantizer.scales
        if w.is_inference() or s.is_inference():
            if not _INFERENCE_TENSOR_WARNED:
                _INFERENCE_TENSOR_WARNED = True
                warnings.warn("QuantLinear weight or scales are inference tensors (created under "
                              "torch.inference_mode()); the weight fold cache is disabled and the "
                              "weight is fake-quantized on every forward. Build/load the model "
                              "outside torch.inference_mode() to enable it.", stacklevel=2)
            return self.weight_quantizer(w)

        key = self._weight_key()
        if self._wq_key == key:
            if not self.keep_source_weights:
                return w
            if self._wq is not None:
                return self._wq
            # keep_source_weights was switched on after an in-place fold with these scales:
            # fall through and cache a (quantization-idempotent) copy of the folded weight.

        if self.keep_source_weights:
            if self._fold_scales is not None and self._scales_changed_since_fold():
                # keep_source_weights was switched on after an in-place fold: the source weight is
                # gone, so new scales cannot be applied to it.
                raise ValueError(
                    "QuantLinear: keep_source_weights was enabled after the weight had already been "
                    "folded in place, and weight_quantizer.scales changed since; the source weights are "
                    "gone. Reload the source weights together with the new scales, or set "
                    "keep_source_weights=True before the first forward.")
            with torch.no_grad():
                self._wq = self.weight_quantizer(w)
            self._wq_key = key
            return self._wq

        self._wq = None
        if self._wq_key is not None and key[:4] == self._wq_key[:4] and self._scales_changed_since_fold():
            raise ValueError(
                "QuantLinear: weight_quantizer.scales changed after the weight was folded in place with "
                "the previous scales; re-quantizing the folded grid would be wrong. Reload the source "
                "weights together with the new scales (load_state_dict with both 'weight' and "
                "'weight_quantizer.scales'), or set keep_source_weights=True before the first forward "
                "to keep an unfolded copy for calibration.")
        with torch.no_grad():
            w.copy_(self.weight_quantizer(w))
            self._fold_scales = s.detach().clone()
        self._wq_key = self._weight_key()
        return w

    def forward(self, x):
        # BLD
        assert not self.training
        # # NOTICE: can use spike_matmul func to substitute the matmul between spikes_int & weight.
        # if self.w_group_size is not None:
        #     spikes_int, vth = dynamic_spikes(x, self.k)
        #     weight = self.weight_quantizer(self.weight).reshape(self.out_features, -1, self.w_group_size)
        #     spikes_int = spikes_int.reshape(*spikes_int.shape[:-1], 1, -1, self.w_group_size)
        #     o =  (spikes_int.float() * weight).sum(-1) # BLOG # group wise matmul.
        #     o = (o * vth.float()).sum(-1).to(self.weight) # BLO
        # else:
        #     spikes_int, vth = dynamic_spikes(x, self.k)
        #     weight = self.weight_quantizer(self.weight)
        #     o =  spikes_int.float() @ weight # BLO
        #     o = (o * vth.float()).to(self.weight)
        # return o

        spikes_int, vth = dynamic_spikes(x, self.k, self.spike_roundtrip)
        x = (spikes_int * vth).to(x.dtype)
        weight = self.quantized_weight()
        out = torch.nn.functional.linear(x, weight, self.bias)
        return out

class Quantizer(nn.Module):
    def __init__(self, in_features: int, out_features: int, w_group_size=None):
        super().__init__()
        
        self.out_features = out_features
        self.in_features = in_features
        self.w_group_size = w_group_size
        
        if w_group_size is None:
            shape = (out_features, 1)
        else:
            shape = (out_features, in_features // w_group_size, 1)
        self.register_buffer('scales', torch.ones(shape))
        # using sym quant for simplicity
        self.register_buffer('zeros', None)

    def forward(self, weight):
        # BLD
        assert not self.training
        org_type = weight.dtype
        if self.w_group_size is not None:
            weight = weight.reshape(self.out_features, -1, self.w_group_size)
            weight = (weight / self.scales).round() * self.scales
            return weight.reshape(self.out_features, self.in_features).to(org_type)
        else:
            weight = (weight / self.scales).round() * self.scales
            return weight.to(org_type)
