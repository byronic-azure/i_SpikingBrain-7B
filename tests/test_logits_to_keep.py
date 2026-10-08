"""Fix #3: lm_head only over the positions that are needed."""
import torch

from conftest import tiny_model


def test_forward_logits_to_keep(pkg):
    model = tiny_model(pkg)
    ids = torch.randint(3, 97, (2, 10))
    with torch.no_grad():
        full = model(ids).logits
        assert full.shape == (2, 10, 97)  # default (0) keeps every position
        torch.testing.assert_close(model(ids, logits_to_keep=1).logits, full[:, -1:], rtol=0, atol=1e-6)
        torch.testing.assert_close(model(ids, num_logits_to_keep=3).logits, full[:, -3:], rtol=0, atol=1e-6)
        idx = torch.tensor([0, 4, 9])
        torch.testing.assert_close(model(ids, logits_to_keep=idx).logits, full[:, idx], rtol=0, atol=1e-6)
        # the loss needs every position, so labels disable slicing
        out = model(ids, labels=ids, logits_to_keep=1)
        assert out.logits.shape == full.shape and torch.isfinite(out.loss)


def test_generate_only_projects_last_position(pkg):
    model = tiny_model(pkg)
    ids = torch.randint(3, 97, (2, 12))
    mask = torch.ones_like(ids)

    seen = []
    handle = model.lm_head.register_forward_hook(lambda m, i, o: seen.append(i[0].shape[1]))
    with torch.no_grad():
        out = model.generate(ids, attention_mask=mask, max_new_tokens=6, do_sample=False, eos_token_id=None)
    handle.remove()
    assert seen[0] == 1, f"prefill projected {seen[0]} positions through lm_head"

    # a hand-written cached greedy loop that projects every position must pick the same tokens
    ref, step_ids, pkv = ids, ids, None
    with torch.no_grad():
        for _ in range(6):
            o = model(step_ids, attention_mask=torch.ones_like(ref), past_key_values=pkv, use_cache=True)
            assert o.logits.shape[1] == step_ids.shape[1]
            pkv, step_ids = o.past_key_values, o.logits[:, -1].argmax(-1, keepdim=True)
            ref = torch.cat([ref, step_ids], 1)
    assert torch.equal(out, ref)
