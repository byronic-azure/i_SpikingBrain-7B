import os

import torch
import torch.nn as nn

try:
    from .Int2Spike.neuron import spike_fake_quant, SpikeCountBitwiseNode, spike_matmul
    spike_is_available = True
except Exception as e:
    print('need https://github.com/BICLab/Int2Spike repo to do fake int2spike, ', e)
    spike_is_available = False

# The bidirectional bitwise spike encode->decode in Int2Spike is an exact identity on
# integer-valued inputs: T = ceil(log2(max|x| + 1)) bits always suffice, the decode
# sums powers of two, and every partial sum of the set bits of an fp32 integer is
# itself exactly representable in fp32. It only loses information outside its own
# domain (non-finite or |x| >= 2**63 values, where the original path raises or is
# undefined). Running it costs ~T extra fp32/int64 passes and 3+ host syncs per
# linear layer, so by default we skip it. Set W8ASPIKE_SPIKE_ROUNDTRIP=1 (or pass
# spike_roundtrip=True) to run the explicit spike path, e.g. to collect firing-rate
# or sparsity statistics.
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

class QuantLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, w_group_size=128, dynamic_sfr=3.0):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)

        self.k = dynamic_sfr
        self.w_group_size = w_group_size
        self.weight_quantizer = Quantizer(in_features, out_features, w_group_size)
        self.spike_roundtrip = None  # None -> follow the module-level SPIKE_ROUNDTRIP default
        self._wq_key = None

    def _weight_key(self):
        w, s = self.weight, self.weight_quantizer.scales
        return (w.data_ptr(), w._version, w.device, w.dtype, s.data_ptr(), s._version)

    def quantized_weight(self):
        """Fake-quantize the weight once and fold the result into ``self.weight``.

        Folding in place keeps memory flat (no second 7B-sized copy). The fold is
        redone whenever the weight or scales storage changes (``load_state_dict``,
        ``.to()``, any in-place write), tracked by data pointer and version counter.
        Re-quantizing already-folded weights is exact as long as |W / scale| <= 255
        (any int8 grid) in bf16, and in fp16/fp32.
        """
        if self._wq_key != self._weight_key():
            with torch.no_grad():
                self.weight.copy_(self.weight_quantizer(self.weight))
            self._wq_key = self._weight_key()
        return self.weight

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
