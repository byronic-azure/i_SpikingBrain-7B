# SpikingBrain-7B — Performance Report

**Scope:** static analysis of the inference hot paths in `hf_7B_model/`, `vllm_hymeta/` and `W8ASpike/`.
**Method:** I read the code paths and derived memory and bandwidth figures from the shipped config (`W8ASpike/config.json`: 28 layers alternating GLA/SWA, 28 query heads, 4 KV heads, head_dim 128, window 4096, vocab 152 064). No GPU was available, so I measured nothing on hardware. Treat the figures as analytical upper and lower bounds, not benchmarks. One claim was checked numerically: the spike encode/decode identity in §4.2, using a numpy replica of the Int2Spike code.

## Summary: ranked opportunities

| # | Area | Issue | Expected impact | Effort |
|---|------|-------|-----------------|--------|
| 1 | W8ASpike | Weights are fake-quantized again on **every** forward call | ~5–7× slower decode than bf16 | S |
| 2 | W8ASpike | Spike encode→decode is an exact identity but costs T≈8–10 passes in fp32/int64, plus 3 host syncs per linear layer | Large activation overhead; OOM risk on long prompts | S |
| 3 | HF | `lm_head` runs over the **whole** prompt | 9.3 GiB at a 32k prompt, 37 GiB at 128k (bf16) | S |
| 4 | HF + vLLM | `repeat_kv` materializes K/V/gate 7× (GQA 28/4) before kernels | 7× K/V traffic in SWA layers; 7× k/v/gk traffic in GLA layers | S–M |
| 5 | HF | SWA cache uses `roll` + `cat` on every decode step | 224 MiB copied per token per sequence once the window is full; O(n²) copying before that | M |
| 6 | HF | `_upad_input` plus a CPU `torch.ones` runs per SWA layer per token | 14 H2D copies and ≥14 device syncs per token | S |
| 7 | vLLM | Prefill loops over requests in Python and syncs on GPU scalars | 3+ syncs × requests × 14 layers per step | M |
| 8 | vLLM | GLA recurrent state is stored in **bf16** | Precision drift over long contexts; 3 GiB at `max_num_seqs=256` | S (trade-off) |

---

## 1. Model shape (the numbers that drive everything)

| Quantity | Value |
|---|---|
| GLA (linear-attention) layers | 14 (even indices) |
| SWA (sliding-window FlashAttention) layers | 14 (odd indices) |
| GQA ratio | 28 / 4 = **7** |
| GLA recurrent state per sequence | 14 × 28 × 128 × 128 → **12.25 MiB** bf16 (24.5 MiB fp32) |
| SWA KV per sequence (window full) | 14 × 2 × 4 × 4096 × 128 → **112 MiB** bf16 |
| …after `repeat_kv` | **784 MiB** |

Decode at batch 1 is bound by weight reads (~15 GB per token). Every per-sequence overhead below scales **linearly with batch size**. At batch ≥16, the SWA-layer copies in §2.2–2.3 exceed the weight traffic.

---

## 2. HuggingFace path (`hf_7B_model/`)

### 2.1 Full-sequence logits during prefill (#3)
`hf_7B_model/modeling_gla_swa.py:420` computes `self.lm_head(hidden_states)` for every position. `generate()` only needs the last one.

| Prompt | Logits (bf16) |
|---|---|
| 4k | 1.16 GiB |
| 32k | 9.28 GiB |
| 128k | 37.1 GiB |

This is often the **dominant allocation** at long context, and it undercuts the model's main selling point (linear-cost long context).
**Fix:** add a `logits_to_keep` / `num_logits_to_keep` argument, as upstream Llama does, and slice `hidden_states[:, -k:]` before `lm_head` when `labels is None`. For training, use fla's `FusedLinearCrossEntropyLoss` so the full logits are never materialized.

### 2.2 `repeat_kv` before FlashAttention (#4)
`window_attention.py:176-177` expands K/V from 4 to 28 heads. `flash_attn_func` and `flash_attn_varlen_func` support GQA natively (`nheads_k` divides `nheads`).
**Fix:** delete the two `repeat_kv` calls. This removes 672 MiB of writes per decode token per sequence (784 − 112), and 7× the K/V memory during prefill.

### 2.3 Sliding-window cache: `roll` and `cat` (#5)
`cache.py:99-100` carries the comment *"DO NOT allocate new memory"*, but `Tensor.roll` **always allocates** a new tensor. Once the window is full, every decode step copies the entire 4096-token K and V for all 14 SWA layers: 224 MiB read+write per token per sequence. Before the window fills, `torch.cat` (`cache.py:106`) reallocates the growing cache every step, which is O(n²) copying.
**Fix:** preallocate a `[b, kv_heads, W, d]` ring buffer and write in place at `pos % W`. RoPE is applied before caching, and a decode query attends to the whole window, so order doesn't matter for a single-token query. For multi-token chunks, pass the buffer through `flash_attn_with_kvcache(cache_seqlens=…)`, which already implements this.

### 2.4 Per-layer unpadding and host transfer (#6)
`window_attention.py:196-199` calls `_upad_input` in every SWA layer on every forward pass. When `attention_mask` is `None`, it also builds `torch.ones(...)` **on the CPU** and copies it to the GPU. `unpad_input` uses `nonzero` and `.item()`, which forces a device sync. With 14 SWA layers, that is at least 14 syncs and up to 14 H2D copies per token, which also prevents CUDA-graph capture.
**Fix:** compute `indices` and `cu_seqlens` once in `HybridModel.forward` and pass them down. When there is no padding, call `flash_attn_func` directly.

### 2.5 Minor
- `repeat_kv` on k, v and gk before `chunk_gla` / `fused_recurrent_gla` (`gla_attention.py:147-149`) does 7× the reads and writes on three tensors. It can only be removed with a GQA-aware GLA kernel (see §3.2).
- RoPE cos/sin is recomputed in each of the 14 SWA layers (`window_attention.py:161`). Compute it once per forward pass, as upstream Llama does.
- `GLU.forward` launches three separate kernels (silu, mul, down). `swiglu_linear` is imported but unused.

---

## 3. vLLM plugin (`vllm_hymeta/`)

### 3.1 Prefill path: Python loop and host syncs (#7)
`model_for_7B/gla_attention.py:186-215`, per prefill request, per GLA layer:
- `query_start_loc[i]` is a GPU scalar used as a slice bound, which forces an implicit `.item()` sync (twice).
- `if initial_state.isnan().any()` forces another sync.
- `fused_chunk_gla` is launched separately for each request.

`modeling_gla_swa.py:406` adds one more sync per request: `context_lens_tensor[i] == 0` inside a Python `if`.
**Fix:** move `query_start_loc` and `context_lens` to the host once per step (the scheduler already has them as CPU lists). Drop the NaN probe, because `_clear_prefill_cache` already zeroes new slots. Then call varlen `chunk_gla(..., cu_seqlens=…, initial_state=batched_states)`, which newer fla versions support, so one launch covers all prefills.

> ⚠️ **Correctness note:** `GLACacheManager` allocates with `torch.empty`. The `isnan()` guard only catches garbage that happens to be NaN. The real protection is `_clear_prefill_cache`, so make sure that function covers every new-sequence path, including chunked prefill and preemption/recompute.

### 3.2 `repeat_kv` in the decode kernel (#4)
`gla_attention.py:265-267` expands k, v and gk 7× before `my_fused_recurrent_gla`. The Triton kernel (`my_fused_recurrent.py`) already indexes by head, so it can read `head_id // 7` for k/v/g at no cost.
**Fix:** pass separate q and kv strides, load k/v/g at `kv_head = pid_h // GROUPS`, and delete the three `repeat_kv` calls. This removes 3 × 7× tensor materializations per GLA layer per step. (Also, `hidden_states.dim == 4` on line 41 compares a *method* to an int, so that branch is dead.)

### 3.3 Decode kernel shape
Grid `(B, H, D/64)` = `(B, 28, 2)`. Each program loads a 128×64 fp32 state tile, which is correct but tiny. At small B the GPU is under-occupied. Options: `BLOCK_SIZE=32` (4 programs per head), or fuse `g_norm` into the kernel epilogue. That also removes the `transpose().contiguous()` pair (`my_fused_recurrent.py`, the tail of `forward`) and one full read+write of the output.

### 3.4 bf16 recurrent state (#8)
`modeling_gla_swa.py:362` stores the GLA state `S_t = exp(g)·S_{t-1} + kᵀv` in bf16, and the kernel accumulates into it at every token. With 8-bit mantissas, small `kᵀv` increments are lost once `S` grows, which is exactly the long-context regime the model targets. fp32 costs 12.25 MiB more per sequence (+3 GiB at 256 slots). This is worth benchmarking with a long-context needle eval before deciding.

---

## 4. W8ASpike path (`W8ASpike/`)

### 4.1 Weight fake-quant runs on every call (#1)
`quant_linear.py:48` calls `self.weight_quantizer(self.weight)` inside `forward`, which computes `(W / scales).round() * scales` for **every linear layer on every token**. `scales` is fp32, so the result is promoted to fp32. That adds three to four extra weight-sized passes per step (~85 GiB of traffic for 7.6B params, compared with ~15 GiB for a plain bf16 read).
**Fix:** the output is deterministic, so fold it once at load time (`self.weight.data = quantizer(self.weight)`) and drop the quantizer from `forward`. To keep real memory savings as well, store int8 weights with per-group scales and use an int8 GEMM such as Marlin.

### 4.2 Spike round-trip is a numerical no-op (#2)
`dynamic_spikes` (`quant_linear.py:17`) rounds activations to integers, then calls `spike_fake_quant(..., SpikeCountBitwiseNode(is_bidirectional=True))`. That function encodes each integer into T sign-magnitude bits and decodes it back with powers of two. Because T = ⌈log₂(max|x|+1)⌉, the round-trip is **exact**.

> Verified: in a numpy replica of `neuronal_charge`, `neuronal_fire` and `spike_dequant` over 200 random activation batches with outliers, the maximum reconstruction error was **0**, with T between 8 and 10.

Per `QuantLinear` call, that no-op costs:
- `torch.allclose` twice (full passes plus syncs), and `x.min().item()` and `x.max().item()` (two more syncs)
- an int64 copy of the activation (4× bf16 size)
- a `[T, *x.shape]` **fp32** spike tensor (for T=9, that is 18× the bf16 activation size) built in a Python loop of T kernel launches
- a decode multiply-sum

It also has a latent failure: if an activation tensor is ever entirely ≥ 0, `spike_quant` raises `ValueError` ("x_zero is required…"), because `dynamic_spikes` passes no `x_zero`.
**Fix:** for inference throughput, skip `spike_fake_quant`; `spikes_int` is already the exact result. Keep the spike path behind a flag (for example `collect_spike_stats=True`) for firing-rate and sparsity measurements, which are what matter for the neuromorphic-hardware story.

---

## 5. Suggested plan

1. **Quick wins, low risk (one PR):** §4.1 fold weights, §4.2 skip the identity round-trip, §2.1 `logits_to_keep`, §2.2 remove `repeat_kv` before FlashAttention, §2.4 hoist unpadding.
2. **Cache rework:** §2.3 ring-buffer SWA cache, or migrate to `flash_attn_with_kvcache`.
3. **Kernel work:** §3.2 GQA-aware decode kernel, §3.1 varlen batched prefill, §3.3 fused `g_norm` epilogue.
4. **Accuracy experiment:** §3.4 fp32 vs bf16 GLA state on long-context retrieval.

**Benchmark to add before merging any of these:** `run_model/` has no timing harness. Add a script that reports TTFT and tokens/s for prompts of {4k, 32k, 128k} × batch sizes {1, 8, 32}, plus peak memory (`torch.cuda.max_memory_allocated`). Run it before and after each change so the impact claims above become measured numbers.
