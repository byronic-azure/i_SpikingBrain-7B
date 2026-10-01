"""Fix #4: GQA without materializing K/V (and the GLA gate) once per query head."""
import importlib.util
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

import cpu_stubs
from conftest import ROOT, load_pkg


def _repeat_kv(x, n_rep):
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _reference_swa(attn, x, window_left, prefix=None):
    """Old code path: explicit repeat_kv, then causal sliding-window attention."""
    b, L, _ = x.shape
    q = attn.q_proj(x).view(b, L, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(b, L, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(b, L, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    pos = torch.arange(L)[None]
    mod = sys.modules[type(attn).__module__]
    cos, sin = attn.rotary_emb(v, pos)
    q, k = mod.apply_rotary_pos_emb(q, k, cos, sin)
    k, v = _repeat_kv(k, attn.num_key_value_groups), _repeat_kv(v, attn.num_key_value_groups)
    i, j = torch.arange(L)[:, None], torch.arange(L)[None]
    mask = (j <= i) & (j >= i - window_left)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return attn.o_proj(o.transpose(1, 2).reshape(b, L, -1))


@pytest.mark.parametrize("q_len", [1, 5, 11])
def test_flash_attention_layer_without_repeat_kv(pkg, q_len):
    if pkg == "W8ASpike":
        pytest.skip("W8ASpike also quantizes q/k/v; its layer is covered by the decode-vs-prefill test")
    modeling, _ = load_pkg(pkg)
    torch.manual_seed(0)
    attn = modeling.FlashAttention(hidden_size=256, num_heads=8, num_key_value_heads=2, sliding_window=4,
                                   layer_idx=0).eval()
    x = torch.randn(2, q_len, 256)
    with torch.no_grad():
        out, _, _ = attn(x, attention_mask=torch.ones(2, q_len), position_ids=torch.arange(q_len)[None])
        ref = _reference_swa(attn, x, window_left=4)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_flash_attention_decode_with_cache_matches_prefill(pkg):
    """Decoding token-by-token through the (un-repeated) KV cache equals one-shot prefill."""
    modeling, _ = load_pkg(pkg)
    Cache = importlib.import_module(pkg + ".cache").Cache
    torch.manual_seed(0)
    attn = modeling.FlashAttention(hidden_size=256, num_heads=8, num_key_value_heads=2, sliding_window=64,
                                   layer_idx=0).eval()
    x = torch.randn(1, 9, 256)
    with torch.no_grad():
        full, _, _ = attn(x, attention_mask=torch.ones(1, 9), position_ids=torch.arange(9)[None])
        cache = Cache()
        attn(x[:, :5], attention_mask=torch.ones(1, 5), position_ids=torch.arange(5)[None], past_key_values=cache)
        steps = []
        for t in range(5, 9):
            o, _, cache = attn(x[:, t:t + 1], attention_mask=torch.ones(1, t + 1),
                               position_ids=torch.tensor([[t]]), past_key_values=cache)
            steps.append(o)
    # the cache must hold num_key_value_heads, not num_heads
    assert cache[0]["attn_state"][0].shape[1] == 2
    torch.testing.assert_close(torch.cat(steps, 1), full[:, 5:], rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------ SDPA path

def _sdpa_module():
    load_pkg("hf_7B_model")
    return importlib.import_module("hf_7B_model.window_attention_sdpa")


def _sdpa_reference(attn, q, k, v, attention_mask, bsz, Q):
    """Original SDPA code path (repeat_kv + mask over all query heads)."""
    k, v = _repeat_kv(k, attn.num_key_value_groups), _repeat_kv(v, attn.num_key_value_groups)
    K = k.shape[2]
    swa = attn._build_sliding_causal_mask(Q, K, attn.sliding_window, device=q.device).view(1, 1, Q, K)
    m = swa | (attention_mask[:, -K:] == 0).view(bsz, 1, 1, K) if attention_mask is not None else swa
    m = torch.where(m[:, :, -Q:, :], -torch.inf, 0.0).to(q.dtype)
    o = torch.nan_to_num(F.scaled_dot_product_attention(q, k, v, attn_mask=m), nan=0.0)
    return attn.o_proj(o.transpose(1, 2).reshape(bsz, Q, -1))


@pytest.mark.parametrize("q_len,pad", [(1, False), (1, True), (7, True), (3000, False)])
def test_sdpa_gqa_fold_matches_repeat_kv(q_len, pad, monkeypatch):
    mod = _sdpa_module()
    torch.manual_seed(0)
    hidden, heads, kv_heads = (256, 8, 2) if q_len < 3000 else (64, 4, 1)  # 3000 >= 2*Hkv*D -> repeat_kv branch
    attn = mod.FlashAttention(hidden_size=hidden, num_heads=heads, num_key_value_heads=kv_heads,
                              sliding_window=5, layer_idx=0).eval()
    Cache = importlib.import_module("hf_7B_model.cache").Cache
    bsz, prefix = 2, 12 if q_len < 3000 else 0
    cache = Cache()
    mask = torch.ones(bsz, prefix + q_len)
    if pad:
        mask[1, :3] = 0
    with torch.no_grad():
        if prefix:
            attn(torch.randn(bsz, prefix, hidden), attention_mask=mask[:, :prefix],
                 position_ids=torch.arange(prefix)[None], past_key_values=cache)
        x = torch.randn(bsz, q_len, hidden)
        pos = torch.arange(prefix, prefix + q_len)[None]

        captured = {}
        real_sdpa = F.scaled_dot_product_attention

        def spy(q, k, v, **kw):
            captured["k_heads"] = k.shape[1]
            return real_sdpa(q, k, v, **kw)

        monkeypatch.setattr(mod.F, "scaled_dot_product_attention", spy)
        out, _, cache = attn(x, attention_mask=mask, position_ids=pos, past_key_values=cache if prefix else None)
        monkeypatch.setattr(mod.F, "scaled_dot_product_attention", real_sdpa)

        # rebuild the post-cache q/k/v exactly like the module does, then run the old path
        q = attn.q_proj(x).view(bsz, q_len, heads, -1).transpose(1, 2)
        k = attn.k_proj(x).view(bsz, q_len, kv_heads, -1).transpose(1, 2)
        v = attn.v_proj(x).view(bsz, q_len, kv_heads, -1).transpose(1, 2)
        cos, sin = attn.rotary_emb(v, pos)
        q, k = mod.apply_rotary_pos_emb(q, k, cos, sin)
        if prefix:
            k, v = cache[0]["attn_state"]
        ref = _sdpa_reference(attn, q, k, v, mask, bsz, q_len)

    expect_fold = q_len < 2 * kv_heads * (hidden // heads)
    assert captured["k_heads"] == (kv_heads if expect_fold else heads)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# ------------------------------------------------------------------ vLLM decode kernel

def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fused_recurrent():
    pytest.importorskip("triton")
    # conftest.py sets TRITON_INTERPRET=1 before triton is first imported, so this runs on CPU.
    return _load_file("_vllm_my_fused_recurrent", os.path.join(ROOT, "vllm_hymeta/model_for_7B/my_fused_recurrent.py"))


def _reference_decode(q, k, v, g, cache, slots, scale):
    """Old path: repeat k/v/g to all query heads, then the recurrent update per slot."""
    H = q.shape[1]
    k, v, g = (_repeat_kv(t, H // t.shape[1]) for t in (k, v, g))
    cache = cache.clone()
    out = torch.zeros(q.shape[0], H, q.shape[-1])
    for b, s in enumerate(slots.tolist()):
        if s == -1:
            continue
        S = cache[s].float() * g[b, :, 0].float().exp()[..., None] + k[b, :, 0, :, None].float() * v[b, :, 0, None, :].float()
        cache[s] = S.to(cache.dtype)
        out[b] = ((q[b, :, 0].float() * scale)[..., None] * S).sum(-2)
    return out, cache


@pytest.mark.parametrize("H,H_kv", [(8, 2), (8, 8), (4, 1)])
def test_vllm_decode_kernel_gqa(fused_recurrent, H, H_kv):
    torch.manual_seed(0)
    B, D, n_slots = 3, 128, 5
    q = torch.randn(B, H, 1, D)
    k, v = torch.randn(B, H_kv, 1, D), torch.randn(B, H_kv, 1, D)
    g = F.logsigmoid(torch.randn(B, H_kv, 1, D)) / 16
    cache = torch.randn(n_slots, H, D, D) * 0.1
    slots = torch.tensor([4, -1, 1], dtype=torch.int32)
    ref_out, ref_cache = _reference_decode(q, k, v, g, cache, slots, D ** -0.5)

    out = fused_recurrent.my_fused_recurrent_gla(q, k, v, g, kv_caches=cache, slot_idx=slots)
    live = slots != -1
    torch.testing.assert_close(out[live], ref_out[live], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cache, ref_cache, rtol=1e-5, atol=1e-5)


# ------------------------------------------------------------------ vLLM GLA layer wiring

@pytest.fixture(scope="module")
def vllm_gla(fused_recurrent):
    """Load vllm_hymeta's gla_attention.py against stub vllm modules (layer logic only, no engine)."""
    names = ["vllm", "vllm.forward_context", "vllm.attention", "vllm.distributed", "vllm.distributed.parallel_state",
             "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.layernorm",
             "vllm.model_executor.layers.linear", "vllm.model_executor.layers.quantization",
             "vllm.model_executor.layers.quantization.base_config", "vllm.model_executor.models",
             "vllm.model_executor.models.constant_size_cache"]
    saved = {n: sys.modules.get(n) for n in names}
    for n in names:
        sys.modules[n] = types.ModuleType(n)
    vm = sys.modules
    vm["vllm.forward_context"].get_forward_context = None
    vm["vllm.attention"].AttentionMetadata = object
    vm["vllm.distributed.parallel_state"].get_tensor_model_parallel_rank = lambda: 0
    vm["vllm.distributed.parallel_state"].get_tensor_model_parallel_world_size = lambda: 1
    vm["vllm.model_executor.layers.layernorm"].RMSNorm = cpu_stubs.RMSNorm
    for n in ("QKVParallelLinear", "ReplicatedLinear", "RowParallelLinear"):
        setattr(vm["vllm.model_executor.layers.linear"], n, object)
    vm["vllm.model_executor.layers.quantization.base_config"].QuantizationConfig = object
    vm["vllm.model_executor.models.constant_size_cache"].ConstantSizeCache = object

    pkg = types.ModuleType("_vllm_m7b")
    pkg.__path__ = [os.path.join(ROOT, "vllm_hymeta/model_for_7B")]
    sys.modules["_vllm_m7b"] = pkg
    sys.modules["_vllm_m7b.my_fused_recurrent"] = fused_recurrent
    try:
        yield importlib.import_module("_vllm_m7b.gla_attention")
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


def test_vllm_prefill_and_decode_mix_matches_full_repeat(vllm_gla):
    """Mixed batch (2 prefills + 2 decodes) with un-repeated k/v/gk equals the old repeat-everything path."""
    torch.manual_seed(0)
    H, H_kv, D = 8, 2, 128
    layer = vllm_gla.GatedLinearAttention.__new__(vllm_gla.GatedLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.num_key_value_groups = H // H_kv
    layer.g_norm = cpu_stubs.RMSNorm(D, eps=1e-6)

    lens = [5, 3]
    n_prefill_tok, n_decode = sum(lens), 2
    N = n_prefill_tok + n_decode
    q = torch.relu(torch.randn(N, H, D))
    k = torch.relu(torch.randn(N, H_kv, D))
    v = torch.randn(N, H_kv, D)
    gk = F.logsigmoid(torch.randn(N, H_kv, D)) / 16
    md = types.SimpleNamespace(num_prefills=2, query_start_loc=torch.tensor([0, 5, 8, 9, 10]),
                               num_decode_tokens=n_decode, num_prefill_tokens=n_prefill_tok)
    slots = torch.tensor([3, 0, 2, 1])
    cache0 = torch.randn(4, H, D, D) * 0.1
    cache0[3] = 0  # fresh prefill slot

    cache_new = cache0.clone()
    out_new = layer._prefill_and_mix_infer(q, k, v, gk, cache_new, slots, md)

    cache_old = cache0.clone()
    G = layer.num_key_value_groups
    kr, vr, gr = (vllm_gla.repeat_kv(t, G) for t in (k, v, gk))
    # old path: k/v/gk repeated up front; with n_rep=1 the per-slice repeat_kv is a no-op
    layer.num_key_value_groups = 1
    out_old = layer._prefill_and_mix_infer(q, kr, vr, gr, cache_old, slots, md)

    torch.testing.assert_close(out_new, out_old, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cache_new, cache_old, rtol=1e-5, atol=1e-5)


def test_repeat_kv_4d_branch_is_live(vllm_gla):
    x = torch.randn(2, 3, 4, 5)
    assert torch.equal(vllm_gla.repeat_kv(x, 2), _repeat_kv(x, 2))
