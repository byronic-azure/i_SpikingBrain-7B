"""The benchmark harness itself runs (CPU, tiny model); real numbers need a GPU."""
import os
import sys

import torch

from conftest import ROOT, tiny_model

sys.path.insert(0, os.path.join(ROOT, "run_model"))
import bench_decode  # noqa: E402


def test_run_case_cpu(pkg):
    model = tiny_model(pkg)
    r = bench_decode.run_case(model, batch_size=2, prompt_len=10, new_tokens=5, device="cpu")
    assert r["new_tokens"] == 5 and r["ttft_ms"] > 0 and r["decode_ms_median"] > 0


def test_random_init_from_config(tmp_path):
    from conftest import TINY, load_pkg
    _, config = load_pkg("W8ASpike")
    path = tmp_path / "config.json"
    config.GLAswaConfig(**TINY).to_json_file(path)
    model = bench_decode.load_model("w8aspike", config_path=str(path), dtype=torch.float32, device="cpu")
    assert next(model.parameters()).dtype == torch.float32
    assert bench_decode.run_case(model, 1, 6, 3, "cpu")["new_tokens"] == 3
