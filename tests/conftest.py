import importlib
import os
import sys

# Run Triton kernels through the CPU interpreter. It must be set before triton is first imported:
# triton.language's own @jit helpers (tl.sum, ...) are only interpretable when created in this mode.
os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cpu_stubs  # noqa: E402

cpu_stubs.install()

TINY = dict(vocab_size=97, hidden_size=256, num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
            intermediate_size=384, sliding_window=6, max_position_embeddings=256, bos_token_id=1, eos_token_id=2,
            pad_token_id=0)


def load_pkg(pkg):
    modeling = importlib.import_module(pkg + ".modeling_gla_swa")
    config = importlib.import_module(pkg + ".configuration_gla_swa")
    cpu_stubs.patch_model_module(modeling)
    return modeling, config


def calibrate_scales(model):
    """Give every QuantLinear realistic per-group int8 scales (absmax / 127) instead of the all-ones init."""
    from W8ASpike.quant_linear import QuantLinear
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, QuantLinear):
                w = m.weight.float().reshape(m.out_features, -1, m.w_group_size)
                m.weight_quantizer.scales.copy_(w.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127)
                m._wq_key = None


def tiny_model(pkg, seed=0, **overrides):
    modeling, config = load_pkg(pkg)
    torch.manual_seed(seed)
    model = modeling.GLAswaForCausalLM(config.GLAswaConfig(**{**TINY, **overrides})).eval()
    if pkg == "W8ASpike":
        calibrate_scales(model)
    return model


@pytest.fixture(params=["hf_7B_model", "W8ASpike"])
def pkg(request):
    return request.param
