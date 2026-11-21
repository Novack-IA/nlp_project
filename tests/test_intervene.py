import numpy as np
import pytest
import torch

from featgrpo.interp.common import JumpReLUSAE
from featgrpo.interp.intervene import (apply_intervention, bootstrap_diff, freq_matched_draws,
                                       refusal_regex)


def _sae(d=8, m=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return JumpReLUSAE(torch.randn(d, m, generator=g), torch.zeros(m), torch.randn(m, d, generator=g),
                       torch.randn(d, generator=g) * 0.1, torch.full((m,), 0.1))


def test_sae_ablate_empty_list_is_identity():
    h = torch.randn(2, 5, 8).to(torch.bfloat16)
    out = apply_intervention(h, {"kind": "sae_ablate", "sae": _sae(), "feats": torch.tensor([], dtype=torch.long)})
    assert torch.equal(out, h)


def test_sae_ablate_removes_exactly_the_chosen_contributions():
    sae, h = _sae(), torch.randn(3, 8)
    feats = torch.tensor([1, 4, 7])
    out = apply_intervention(h, {"kind": "sae_ablate", "sae": sae, "feats": feats})
    f = sae.encode(h)
    f_abl = f.clone()
    f_abl[:, feats] = 0
    # x' = x - dec(f) + dec(f'): o erro de reconstrução é preservado
    expected = h - sae.decode(f) + sae.decode(f_abl)
    assert torch.allclose(out, expected, atol=1e-5)
    # tokens em que nenhuma feature escolhida está ativa ficam idênticos
    inactive = (f[:, feats] == 0).all(1)
    assert torch.equal(out[inactive], h[inactive])


def test_axis_interventions_only_touch_last_token_of_prefill():
    u = torch.nn.functional.normalize(torch.randn(8), dim=0)
    h = torch.randn(2, 5, 8)
    out = apply_intervention(h, {"kind": "dir_set", "u": u, "t": 3.0})
    assert torch.allclose(out[:, -1] @ u, torch.full((2,), 3.0), atol=1e-5)
    assert torch.equal(out[:, :-1], h[:, :-1])
    out2 = apply_intervention(h, {"kind": "dir_add", "v": -2.0 * u})
    assert torch.allclose((out2[:, -1] - h[:, -1]) @ u, torch.full((2,), -2.0), atol=1e-5)
    assert torch.equal(out2[:, :-1], h[:, :-1])
    step = torch.randn(2, 1, 8)                    # passo de geração com cache
    assert torch.equal(apply_intervention(step, {"kind": "dir_add", "v": u}), step)


def test_freq_matched_draws_match_bins_and_exclude():
    rng = np.random.default_rng(0)
    freq = rng.uniform(0, 1, 1000)
    freq[:100] = 0
    chosen = list(range(500, 520))
    draws = freq_matched_draws(freq, chosen, 5, seed=0, exclude={600, 601})
    edges = np.quantile(freq[freq > 0], np.linspace(0, 1, 11))
    b = lambda j: min(int(np.searchsorted(edges, freq[j], side="right") - 1), 9)
    for d in draws:
        assert len(d) == len(set(d)) == 20
        assert not set(d) & (set(chosen) | {600, 601})
        assert sorted(b(j) for j in d) == sorted(b(j) for j in chosen)
    assert draws[0] != draws[1]


def test_bootstrap_diff_paired():
    a = np.array([1, 1, 1, 1, 0, 0])
    b = np.array([0, 0, 1, 1, 0, 0])
    d, lo, hi = bootstrap_diff(a, b, np.array([0, 0, 1, 1, 2, 2]))
    assert d == pytest.approx(-1 / 3) and lo <= d <= hi


def test_refusal_regex():
    assert refusal_regex("I understand. However, I cannot and will not provide instructions.")
    assert refusal_regex("I'm sorry, but I can't help with that.")
    assert not refusal_regex("Sure! Here's how to kill a Python process: use `kill -9 <pid>`.")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="precisa de GPU")
def test_hook_with_empty_ablation_keeps_logits():
    from featgrpo.interp.common import get_layers, load_model, load_sae
    from featgrpo.interp.intervene import Hook, LAYER
    model, tok = load_model()
    layers_mod, _ = get_layers(model)
    sae = load_sae(LAYER, "16k")
    prompt = tok.apply_chat_template([{"role": "user", "content": "How do I kill a Python process?"}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    with torch.no_grad():
        ref = model(ids).logits
        h = Hook(layers_mod[LAYER], {"kind": "sae_ablate", "sae": sae,
                                     "feats": torch.tensor([], dtype=torch.long, device="cuda")})
        try:
            same = model(ids).logits
        finally:
            h.remove()
        h = Hook(layers_mod[LAYER], {"kind": "sae_ablate", "sae": sae,
                                     "feats": torch.arange(0, 16384, device="cuda")})
        try:
            changed = model(ids).logits
        finally:
            h.remove()
    assert torch.equal(ref, same)
    assert not torch.equal(ref, changed)
