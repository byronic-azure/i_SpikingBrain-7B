"""Decode-latency and peak-memory benchmark for the HF and W8ASpike inference paths.

For each (prompt length, batch size) it runs greedy `generate()` and reports:
  * TTFT        - prefill + first token (ms)
  * decode      - median / p90 latency per generated token after the first (ms) and tokens/s
  * peak mem    - torch.cuda.max_memory_allocated() over the whole call, and the part above the weights

A/B usage: run once on `main` and once on this branch with the same arguments, then compare the JSON.

    # real checkpoint, using the modeling code from this repo (not the copy in the checkpoint dir)
    python run_model/bench_decode.py --impl hf --model-path /ckpt/SpikingBrain-7B \\
        --prompt-lens 4096 32768 --batch-sizes 1 8 --new-tokens 64 --json hf.json

    # no checkpoint needed: random weights from a config.json (latency/memory do not depend on weights)
    python run_model/bench_decode.py --impl w8aspike --config W8ASpike/config.json --prompt-lens 4096

    # W8ASpike with the original spike encode->decode round-trip, to isolate fix #2
    W8ASPIKE_SPIKE_ROUNDTRIP=1 python run_model/bench_decode.py --impl w8aspike ...

Needs a CUDA GPU with flash-attn and flash-linear-attention installed (see requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = {"hf": "hf_7B_model", "w8aspike": "W8ASpike"}


class TokenTimer(StoppingCriteria):
    """Called by generate() once per generated token; records a synchronized timestamp each time."""

    def __init__(self, device):
        self.device = device
        self.stamps = []

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self):
        self._sync()
        self.t0 = time.perf_counter()

    def __call__(self, input_ids, scores, **kwargs):
        self._sync()
        self.stamps.append(time.perf_counter())
        return False


def load_model(impl, model_path=None, config_path=None, dtype=torch.bfloat16, device="cuda"):
    sys.path.insert(0, ROOT)
    pkg = PACKAGES[impl]
    modeling = __import__(pkg + ".modeling_gla_swa", fromlist=["GLAswaForCausalLM"])
    configuration = __import__(pkg + ".configuration_gla_swa", fromlist=["GLAswaConfig"])
    if model_path:
        model = modeling.GLAswaForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    else:
        config = configuration.GLAswaConfig.from_json_file(config_path)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)  # build directly in bf16/fp16 on the device: no fp32 7B copy
        try:
            with torch.device(device):
                model = modeling.GLAswaForCausalLM(config)
        finally:
            torch.set_default_dtype(default_dtype)
        if impl == "w8aspike":
            calibrate_w8_scales(model)
    return model.to(device).eval()


def calibrate_w8_scales(model):
    """Random-init W8ASpike models get per-group int8 scales so the weight grid is realistic."""
    from W8ASpike.quant_linear import QuantLinear
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, QuantLinear):
                w = m.weight.float().reshape(m.out_features, -1, m.w_group_size)
                m.weight_quantizer.scales.copy_(w.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127)


def run_case(model, batch_size, prompt_len, new_tokens, device, seed=0):
    device = torch.device(device)
    g = torch.Generator().manual_seed(seed)
    vocab = model.config.vocab_size
    ids = torch.randint(10, vocab - 10, (batch_size, prompt_len), generator=g).to(device)
    mask = torch.ones_like(ids)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
    timer = TokenTimer(device)
    with torch.inference_mode():
        timer.start()
        model.generate(ids, attention_mask=mask, do_sample=False, max_new_tokens=new_tokens,
                       min_new_tokens=new_tokens, eos_token_id=None, pad_token_id=0,
                       stopping_criteria=StoppingCriteriaList([timer]))
    stamps = [timer.t0] + timer.stamps
    steps = [(b - a) * 1e3 for a, b in zip(stamps[1:], stamps[2:])]
    result = {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "new_tokens": len(timer.stamps),
        "ttft_ms": (stamps[1] - stamps[0]) * 1e3,
        "decode_ms_median": statistics.median(steps) if steps else float("nan"),
        "decode_ms_p90": sorted(steps)[int(0.9 * (len(steps) - 1))] if steps else float("nan"),
    }
    result["decode_tok_s"] = batch_size * 1e3 / result["decode_ms_median"] if steps else float("nan")
    if device.type == "cuda":
        result["peak_mem_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        result["peak_above_weights_gib"] = (torch.cuda.max_memory_allocated(device) - base) / 2**30
    return result


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=sorted(PACKAGES), required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-path", help="checkpoint directory (weights); modeling code comes from this repo")
    src.add_argument("--config", help="config.json for a random-weight model (no checkpoint needed)")
    p.add_argument("--prompt-lens", type=int, nargs="+", default=[4096])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--warmup", type=int, default=1, help="untimed warmup runs (short prompt) before measuring")
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", help="also write results to this file")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("No CUDA device: this benchmark needs a GPU (the fla / flash-attn kernels are CUDA-only).")

    model = load_model(args.impl, args.model_path, args.config, getattr(torch, args.dtype), args.device)
    for _ in range(args.warmup):
        run_case(model, 1, 128, 8, args.device)

    meta = {"impl": args.impl, "git": git_rev(), "dtype": args.dtype,
            "spike_roundtrip": os.environ.get("W8ASPIKE_SPIKE_ROUNDTRIP", "0"),
            "gpu": torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else "cpu"}
    print(json.dumps(meta))
    header = f"{'batch':>5} {'prompt':>7} {'TTFT ms':>9} {'tok ms p50':>10} {'p90':>8} {'tok/s':>8} {'peak GiB':>9} {'+weights':>9}"
    print(header)
    rows = []
    for prompt_len in args.prompt_lens:
        for bs in args.batch_sizes:
            try:
                r = run_case(model, bs, prompt_len, args.new_tokens, args.device)
            except torch.cuda.OutOfMemoryError:
                r = {"batch_size": bs, "prompt_len": prompt_len, "error": "OOM"}
                print(f"{bs:>5} {prompt_len:>7}  OOM")
                rows.append(r)
                continue
            rows.append(r)
            print(f"{bs:>5} {prompt_len:>7} {r['ttft_ms']:>9.1f} {r['decode_ms_median']:>10.2f} {r['decode_ms_p90']:>8.2f} "
                  f"{r['decode_tok_s']:>8.1f} {r.get('peak_mem_gib', float('nan')):>9.2f} "
                  f"{r.get('peak_above_weights_gib', float('nan')):>9.2f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "results": rows}, f, indent=2)


if __name__ == "__main__":
    main()
