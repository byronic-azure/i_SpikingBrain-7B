"""CPU reference stand-ins for the CUDA-only dependencies (fla, flash_attn).

They let the HF / W8ASpike model code run on CPU with tiny random configs so the
equivalence tests can exercise real model code paths without a GPU. Each stub
mirrors the semantics of the real kernel it replaces, including the head-count
contract: the GLA kernels require equal q/k/v heads (like fla 0.1), while the
flash-attention stubs accept GQA (nheads % nheads_k == 0), like flash_attn 2.x.

Call ``install()`` before importing ``hf_7B_model`` / ``W8ASpike``.
"""
from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- fla stubs

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        if residual is not None:
            x = x + residual
        residual_out = x
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        y = y.to(x.dtype)
        return (y, residual_out) if prenorm else y


class ShortConvolution(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("ShortConvolution is unused by the shipped config (use_short_conv=False)")


class FusedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, inplace_backward=False, **kwargs):
        super().__init__(**kwargs)


def naive_gla(q, k, v, gk, scale=None, initial_state=None, output_final_state=False, **kwargs):
    """Reference gated linear attention: S_t = diag(exp(g_t)) S_{t-1} + k_t^T v_t, o_t = q_t S_t."""
    B, H, L, Dk = q.shape
    # fla 0.1 GLA kernels need one k/v/g head per q head; mirror that contract.
    assert k.shape[1] == H and v.shape[1] == H and gk.shape[1] == H, "GLA kernel needs equal head counts"
    Dv = v.shape[-1]
    scale = Dk ** -0.5 if scale is None else scale
    S = q.new_zeros(B, H, Dk, Dv, dtype=torch.float32) if initial_state is None else initial_state.float().clone()
    o = q.new_empty(B, H, L, Dv, dtype=torch.float32)
    for t in range(L):
        S = S * gk[:, :, t].float().exp()[..., None] + k[:, :, t].float()[..., None] * v[:, :, t].float()[:, :, None, :]
        o[:, :, t] = (q[:, :, t].float() * scale)[..., None].mul(S).sum(-2)
    return o.to(q.dtype), (S if output_final_state else None)


def _passthrough(fn=None, **kwargs):
    return fn if fn is not None else (lambda f: f)


# ---------------------------------------------------------- flash_attn stubs

def _attend(q, k, v, softmax_scale, causal, window_size):
    """q: [Lq, H, D], k/v: [Lk, Hk, D]; flash-attn mask semantics (bottom-right aligned)."""
    Lq, H, _ = q.shape
    Lk, Hk, _ = k.shape
    assert H % Hk == 0, "flash_attn needs nheads % nheads_k == 0"
    k = k.repeat_interleave(H // Hk, dim=1)
    v = v.repeat_interleave(H // Hk, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * softmax_scale
    i = torch.arange(Lq)[:, None] + (Lk - Lq)
    j = torch.arange(Lk)[None, :]
    allowed = torch.ones(Lq, Lk, dtype=torch.bool)
    if causal:
        allowed &= j <= i
    left, right = window_size
    if left >= 0:
        allowed &= j >= i - left
    if right >= 0 and not causal:
        allowed &= j <= i + right
    scores = scores.masked_fill(~allowed, float("-inf"))
    out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
    return out.to(q.dtype)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), **kwargs):
    softmax_scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    return torch.stack([_attend(q[b], k[b], v[b], softmax_scale, causal, window_size) for b in range(q.shape[0])])


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                           dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), **kwargs):
    softmax_scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    outs = []
    for b in range(len(cu_seqlens_q) - 1):
        qs, qe = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        ks, ke = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
        outs.append(_attend(q[qs:qe], k[ks:ke], v[ks:ke], softmax_scale, causal, window_size))
    return torch.cat(outs)


def pad_input(hidden_states, indices, batch, seqlen):
    out = hidden_states.new_zeros(batch * seqlen, *hidden_states.shape[1:])
    out[indices] = hidden_states
    return out.view(batch, seqlen, *hidden_states.shape[1:])


def _unpad_data(mask):
    seqlens = mask.sum(-1, dtype=torch.int32)
    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    cu = F.pad(torch.cumsum(seqlens, 0, dtype=torch.int32), (1, 0))
    return indices, cu, int(seqlens.max())


def upad_input(q, k, v, attention_mask, query_length, *unused):
    """transformers._upad_input, CPU version; accepts both the 5- and 6-argument signatures."""
    indices_k, cu_k, max_k = _unpad_data(attention_mask)
    b, kv_len, hk, d = k.shape
    k = k.reshape(b * kv_len, hk, d)[indices_k]
    v = v.reshape(b * kv_len, hk, d)[indices_k]
    if query_length == kv_len:
        q = q.reshape(b * kv_len, q.shape[2], d)[indices_k]
        indices_q, cu_q, max_q = indices_k, cu_k, max_k
    elif query_length == 1:
        cu_q = torch.arange(b + 1, dtype=torch.int32)
        indices_q = cu_q[:-1].long()
        max_q = 1
        q = q.squeeze(1)
    else:
        indices_q, cu_q, max_q = _unpad_data(attention_mask[:, -query_length:])
        q = q.reshape(b * query_length, q.shape[2], d)[indices_q]
    return q, k, v, indices_q, (cu_q, cu_k), (max_q, max_k)


def swiglu_linear(x, y, weight, bias):
    return F.linear(F.silu(x) * y, weight, bias)


# ------------------------------------------------------------------ install

def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def install():
    if getattr(sys.modules.get("fla"), "_is_cpu_stub", False):
        return
    fla = _module("fla", _is_cpu_stub=True)
    fla.modules = _module("fla.modules", RMSNorm=RMSNorm, ShortConvolution=ShortConvolution,
                          FusedCrossEntropyLoss=FusedCrossEntropyLoss)
    fla.modules.activations = _module("fla.modules.activations", swish=F.silu)
    fla.ops = _module("fla.ops")
    fla.ops.gla = _module("fla.ops.gla", chunk_gla=naive_gla, fused_chunk_gla=naive_gla, fused_recurrent_gla=naive_gla)
    fla.utils = _module("fla.utils", contiguous=_passthrough, autocast_custom_fwd=_passthrough,
                        autocast_custom_bwd=_passthrough)

    fa = _module("flash_attn", flash_attn_func=flash_attn_func, flash_attn_varlen_func=flash_attn_varlen_func)
    fa.flash_attn_interface = _module("flash_attn.flash_attn_interface", flash_attn_func=flash_attn_func,
                                      flash_attn_varlen_func=flash_attn_varlen_func)
    fa.bert_padding = _module("flash_attn.bert_padding", pad_input=pad_input, unpad_input=None,
                              index_first_axis=lambda x, idx: x[idx])

    import transformers.utils
    transformers.utils.is_flash_attn_2_available = lambda: True


def patch_model_module(mod):
    """Point a loaded modeling package at the CPU stubs (names bound with `from x import y`)."""
    pkg = mod.__name__.rsplit(".", 1)[0]
    wa = sys.modules[pkg + ".window_attention"]
    wa._upad_input = upad_input
    wa.flash_attn_func = flash_attn_func
    wa.flash_attn_varlen_func = flash_attn_varlen_func
    wa.pad_input = pad_input
    mod.swiglu_linear = swiglu_linear
