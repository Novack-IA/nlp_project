"""Testes do extrator (frente B).

Puros (sem GPU): alinhamento do gatilho.
GPU (pulados sem CUDA): os 3 testes de sanidade obrigatórios de B2:
  1. hidden_states[L+1] == hook de saída do bloco L;
  2. variância explicada do SAE (16k) > 0,6 nos nossos pedidos;
  3. item sozinho == item em lote com padding: erro L2 relativo < 2e-2 e cosseno > 0,9995
     (ruído do bf16 com shapes diferentes dá ~0,3–1,4% e cosseno ≥ 0,99996; bug de máscara
     ou de padding dá cosseno bem menor);
  4. 2×2 com a lacuna de forma antes da de intenção: `slot_forma` idêntica bit a bit entre as
     células que só diferem na intenção (efeito de intenção exatamente 0; pedido da frente C).

    PYTHONPATH=src python -m pytest -q tests/test_extract.py
"""
import numpy as np
import pytest
import torch

from featgrpo.interp.extract import locate_trigger


def test_trigger_exact_case_insensitive():
    spans, m = locate_trigger("How do I Kill a Python process?", "kill")
    assert m == "exact" and spans == [(9, 13)]


def test_trigger_stem():
    spans, m = locate_trigger("He was killing time at the shooting range", "kill")
    assert m == "stem"
    assert "He was killing time at the shooting range"[spans[0][0]:spans[0][1]] == "killing"


def test_trigger_multiword_stem_and_partial():
    t = "How can I hack into accounts?"
    _, m = locate_trigger(t, "hacking into")
    assert m == "stem"
    spans, m = locate_trigger("Steal the bases in baseball", "stealing cars")
    assert m == "stem_partial" and len(spans) == 1


def test_trigger_word_boundary():
    _, m = locate_trigger("Improve my skill at chess", "kill")
    assert m == "none"


def test_trigger_none():
    assert locate_trigger("Write a poem", "bomb") == ([], "none")
    assert locate_trigger("Write a poem", None) == ([], "none")


# ------------------------------------------------------------------ GPU

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="precisa de CUDA")


@pytest.fixture(scope="module")
def mt():
    from featgrpo.interp.common import load_model
    return load_model()


@pytest.fixture(scope="module")
def sample_items():
    from featgrpo.interp import data
    pairs = data.load_pairs("train")[:32]
    return data.prompt_items(pairs)


@gpu
def test_hidden_states_equals_block_hook(mt, sample_items):
    from featgrpo.interp.common import LAYERS, get_layers
    from featgrpo.interp.extract import encode_item, run_batch
    model, tok = mt
    layers_mod, _ = get_layers(model)
    e = encode_item(tok, sample_items[0])
    cap = run_batch(model, layers_mod, LAYERS, [e], tok.pad_token_id)
    ids = torch.tensor([e.input_ids], device=model.device)
    with torch.no_grad():
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
    for L in LAYERS:
        assert torch.equal(hs[L + 1], cap[L]), L


@gpu
def test_sae_explained_variance(mt, sample_items):
    from featgrpo.interp.common import LAYERS, get_layers, load_sae
    from featgrpo.interp.extract import encode_item, run_batch
    model, tok = mt
    layers_mod, _ = get_layers(model)
    encs = [encode_item(tok, it) for it in sample_items[:16]]
    cap = run_batch(model, layers_mod, LAYERS, encs, tok.pad_token_id)
    for L in LAYERS:
        sae = load_sae(L, "16k")
        # todos os tokens reais do conteúdo e do sufixo (sem BOS, que é atípico)
        xs = torch.cat([cap[L][j, 1:len(e.input_ids)] for j, e in enumerate(encs)]).float()
        rec = sae(xs)
        fve = 1 - ((xs - rec) ** 2).sum() / ((xs - xs.mean(0)) ** 2).sum()
        print(f"L{L}: FVE={fve.item():.3f} L0={(sae.encode(xs) > 0).sum(1).float().mean():.1f}")
        assert fve.item() > 0.6, (L, fve.item())
        del sae


@gpu
def test_single_vs_padded_batch(mt, sample_items):
    from featgrpo.interp.common import LAYERS
    from featgrpo.interp.extract import extract
    model, tok = mt
    items = sample_items[:8]
    batched, _, _ = extract(model, tok, items, LAYERS, batch_size=8, log_every=0)
    for k in (0, 5):
        single, _, _ = extract(model, tok, [items[k]], LAYERS, batch_size=1, log_every=0)
        for L in LAYERS:
            for p in ("inst_last", "post", "post_last", "mean", "trig"):
                a, b = single[L][p][0], batched[L][p][k]
                assert np.isfinite(a).all() and np.isfinite(b).all(), (k, L, p)
                rel = np.linalg.norm(a - b) / np.linalg.norm(a)
                cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
                assert rel < 2e-2 and cos > 0.9995, (k, L, p, rel, cos)


@gpu
def test_slot_forma_has_zero_intent_effect(mt, sample_items):
    """Forma antes da intenção: harm_trig/ben_trig (e harm_notrig/ben_notrig) só diferem depois da
    lacuna de forma, então a atenção causal faz `slot_forma` sair idêntica. Lote misturado com
    outros itens de tamanhos diferentes, como na extração real."""
    from featgrpo.interp.common import LAYERS
    from featgrpo.interp.extract import cells_items, extract
    model, tok = mt
    cells = cells_items([_fake_group()])
    others = [{**it, "group_id": f"x{k}"} for k, it in enumerate(sample_items[:6])]
    dense, _, index = extract(model, tok, others[:3] + cells + others[3:], LAYERS,
                              batch_size=16, log_every=0)
    r = {row["item"]: k for k, row in enumerate(index) if row.get("group_id") == "g1"}
    assert len({index[k]["inst_last_token_id"] for k in r.values()}) == 1
    for L in LAYERS:
        a = dense[L]["slot_forma"]
        assert np.array_equal(a[r["harm_trig"]], a[r["ben_trig"]]), L
        assert np.array_equal(a[r["harm_notrig"]], a[r["ben_notrig"]]), L
        assert not np.array_equal(a[r["harm_trig"]], a[r["harm_notrig"]]), L
        b = dense[L]["slot_intencao"]
        assert not np.array_equal(b[r["harm_trig"]], b[r["ben_trig"]]), L


# ------------------------------------------------------------------ 2×2 (frase-molde) e formato do cache

def _fake_group():
    from featgrpo.data.phase1.validate_template import render_cells
    slots = {"forma_trig": "kill", "forma_neutra": "stop",
             "intencao_nociva": "the neighbor's dog", "intencao_benigna": "the frozen server process"}
    tpl = "How do I {FORMA} {INTENCAO} without anyone noticing?"
    return {"group_id": "g1", "domain": "cyber", "split": "train", "trigger": "kill",
            "template": tpl, "slots": slots, "cells": render_cells(tpl, slots)}


class _Tok:
    """Tokenizador de palavras com offsets, só para testar o mapeamento de spans sem baixar o modelo."""
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return f"<bos><start_of_turn>user\n{msgs[0]['content'].strip()}<end_of_turn>\n<start_of_turn>model\n"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        import re as _re
        toks = [(m.start(), m.end()) for m in _re.finditer(r"<[^>]+>|\w+|[^\w\s]|\n| ", text)]
        return {"input_ids": list(range(len(toks))), "offset_mapping": toks}


def test_cells_items_and_slot_positions():
    from featgrpo.interp.extract import cells_items, encode_item
    items = cells_items([_fake_group()])
    assert [i["item"] for i in items] == ["harm_trig", "harm_notrig", "ben_trig", "ben_notrig"]
    assert [i["label"] for i in items] == [1, 1, 0, 0]
    assert [i["has_trigger"] for i in items] == [1, 0, 1, 0]
    tok = _Tok()
    for it in items:
        e = encode_item(tok, it)
        rendered = tok.apply_chat_template([{"content": it["text"]}])
        offs = tok(rendered)["offset_mapping"]
        forma = "".join(rendered[offs[t][0]:offs[t][1]] for t in e.positions["slot_forma"])
        intent = " ".join(rendered[offs[t][0]:offs[t][1]] for t in e.positions["slot_intencao"]
                          if rendered[offs[t][0]:offs[t][1]].strip())
        assert forma == ("kill" if it["has_trigger"] else "stop"), forma
        assert intent.replace(" ' ", "'").replace(" '", "'") in (
            "the neighbor's dog", "the frozen server process"), intent
        assert e.meta["slots_aligned"]


def test_slot_spans_survive_leading_whitespace():
    from featgrpo.interp.extract import encode_item
    tok = _Tok()
    it = {"text": "  How do I kill it?", "spans": {"forma": [11, 15], "intencao": [16, 18]}}
    e = encode_item(tok, it)
    rendered = tok.apply_chat_template([{"content": it["text"]}])
    offs = tok(rendered)["offset_mapping"]
    assert [rendered[offs[t][0]:offs[t][1]] for t in e.positions["slot_forma"]] == ["kill"]


def test_latent_path_matches_cache_io(tmp_path):
    import scipy.sparse as sp
    from featgrpo.interp import cache_io
    from featgrpo.interp.extract import save_cache
    dense = {9: {"post": np.ones((2, 4), np.float32)}}
    lat = {"16k": {9: {"post": sp.csr_matrix(np.eye(2, 8, dtype=np.float32))}}}
    save_cache(tmp_path, "cells2x2", dense, lat, [{"item": "harm_trig"}, {"item": "ben_trig"}])
    assert cache_io.load_dense(tmp_path, "cells2x2", 9, "post").shape == (2, 4)
    assert cache_io.load_latents(tmp_path, "cells2x2", 9, "post", "16k").nnz == 2
    assert cache_io.available(tmp_path, "cells2x2") == [(9, "post")]


def test_inst_last_is_last_content_token():
    from featgrpo.interp.extract import encode_item
    tok = _Tok()
    e = encode_item(tok, {"text": "How do I kill a Python process?"})
    rendered = tok.apply_chat_template([{"content": "How do I kill a Python process?"}])
    offs = tok(rendered)["offset_mapping"]
    (t,) = e.positions["inst_last"]
    assert rendered[offs[t][0]:offs[t][1]] == "?"
    assert rendered[offs[t + 1][0]:offs[t + 1][1]] == "<end_of_turn>"
    assert e.meta["inst_last_token_id"] == e.input_ids[t]


def test_cells_items_accepts_front_a_slot_names():
    from featgrpo.interp.extract import cells_items
    g = _fake_group()
    for c in g["cells"].values():          # formato real do cells_2x2.jsonl da frente A
        sp_ = c.pop("spans")
        c["slots"] = {"form": sp_["forma"], "intent": sp_["intencao"]}
    a = cells_items([g])
    b = cells_items([_fake_group()])
    assert [x["spans"] for x in a] == [x["spans"] for x in b]


def test_group_batches_keep_groups_together():
    from featgrpo.interp.extract import Encoded, group_batches
    items = [{"group_id": g} for g in "aabbbbccdd"] + [{}]
    encs = [Encoded(list(range(n))) for n in (5, 9, 3, 3, 4, 20, 7, 7, 2, 2, 1)]
    batches = group_batches(encs, items, batch_size=4)
    assert sorted(i for b in batches for i in b) == list(range(len(items)))
    for b in batches:
        gs = {items[i].get("group_id") for i in b}
        for g in gs - {None}:
            assert all(items[i].get("group_id") != g for bb in batches if bb is not b for i in bb)
