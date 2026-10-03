"""End-to-end check that this tree computes exactly what a git ref (default: main) computes.

Builds the same tiny random HF and W8ASpike models from both code trees (CPU stubs for the
CUDA kernels, see cpu_stubs.py), then compares full-sequence logits and greedy generations.

    python tests/compare_with_ref.py            # vs main
    python tests/compare_with_ref.py --ref HEAD~1
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def collect(root, out):
    os.environ.setdefault("TRITON_INTERPRET", "1")
    sys.path.insert(0, root)
    sys.path.insert(1, HERE)
    import cpu_stubs
    cpu_stubs.install()
    import importlib
    import torch
    # same tiny config as conftest.TINY (not imported: conftest puts this tree on sys.path)
    TINY = dict(vocab_size=97, hidden_size=256, num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
                intermediate_size=384, max_position_embeddings=256, bos_token_id=1, eos_token_id=2, pad_token_id=0)

    res = {}
    for pkg in ["hf_7B_model", "W8ASpike"]:
        modeling = importlib.import_module(pkg + ".modeling_gla_swa")
        config = importlib.import_module(pkg + ".configuration_gla_swa")
        assert modeling.__file__.startswith(root), modeling.__file__
        cpu_stubs.patch_model_module(modeling)
        for window in (6, 64):
            torch.manual_seed(0)
            model = modeling.GLAswaForCausalLM(config.GLAswaConfig(**{**TINY, "sliding_window": window})).eval()
            if pkg == "W8ASpike":
                ql = importlib.import_module("W8ASpike.quant_linear")
                with torch.no_grad():
                    for m in model.modules():
                        if isinstance(m, ql.QuantLinear):
                            w = m.weight.float().reshape(m.out_features, -1, m.w_group_size)
                            m.weight_quantizer.scales.copy_(w.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127)
            ids = torch.randint(3, 97, (2, 13), generator=torch.Generator().manual_seed(1))
            gen = dict(do_sample=False, max_new_tokens=10, eos_token_id=None)
            with torch.no_grad():
                res[f"{pkg}/w{window}/logits"] = model(ids, attention_mask=torch.ones_like(ids)).logits
                res[f"{pkg}/w{window}/generate_bs2"] = model.generate(ids, attention_mask=torch.ones_like(ids), **gen)
                res[f"{pkg}/w{window}/generate_bs1"] = model.generate(ids[:1], **gen)
    torch.save(res, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="main")
    p.add_argument("--_collect", nargs=2, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args._collect:
        return collect(*args._collect)

    import torch
    with tempfile.TemporaryDirectory() as tmp:
        ref_root = os.path.join(tmp, "ref")
        os.makedirs(ref_root)
        archive = subprocess.run(["git", "-C", ROOT, "archive", args.ref, "hf_7B_model", "W8ASpike"],
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", ref_root], input=archive, check=True)
        outs = {}
        for name, root in (("ref", ref_root), ("tree", ROOT)):
            outs[name] = os.path.join(tmp, name + ".pt")
            proc = subprocess.run([sys.executable, __file__, "--_collect", root, outs[name]],
                                  capture_output=True, text=True)
            if proc.returncode:
                sys.exit(f"collecting outputs from {name} failed:\n{proc.stderr}")
        ref, tree = torch.load(outs["ref"]), torch.load(outs["tree"])

    ok = True
    for key in ref:
        same = torch.equal(ref[key], tree[key])
        ok &= same
        extra = f"  max|diff|={(ref[key] - tree[key]).abs().max().item():.3g}" if key.endswith("logits") else ""
        print(f"{'OK  ' if same else 'DIFF'} {key}{extra}")
    print(f"\n{'bitwise identical to' if ok else 'DIFFERS from'} {args.ref}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
