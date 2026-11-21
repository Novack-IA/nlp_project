"""Testes de probes.py e classifiers.py com um cache sintético no formato do AGENTS.md (seção 4).

O cache tem sinais plantados: uma direção (e features) de intenção e outra de gatilho.
Os testes checam que o código recupera cada sinal onde ele está, e só lá.
"""
import json

import numpy as np
import pytest
import scipy.sparse as sp

from featgrpo.data.phase1.validate_template import CELLS
from featgrpo.interp import cache_io, classifiers, probes

D, NF, L, POS = 24, 40, 5, "post"
DOMAINS = ("bias", "cyber", "privacy", "violence")
INTENT_FEATS, LEX_FEATS = list(range(5)), list(range(5, 10))


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("acts")
    rng = np.random.default_rng(0)
    triggers = [f"trig{i}" for i in range(30)]
    cell_rows, pair_rows, Xc, Zc, Xp = [], [], [], [], []
    for gi in range(240):
        split = "train" if gi < 144 else "val" if gi < 192 else "test"
        trig = triggers[rng.integers(0, 30 if split == "train" else 40) % 30] if split == "train" \
            else (triggers[rng.integers(0, 30)] if rng.random() < 0.5 else f"novo{gi}")
        for cell in CELLS:
            h, t = int(cell.startswith("harm")), int(cell.endswith("_trig"))
            cell_rows.append({"group_id": f"g{gi}", "item": cell, "split": split, "domain": DOMAINS[gi % 4],
                              "trigger": trig, "n_tokens": 20, "trigger_aligned": True})
            x = rng.normal(size=D)
            x[0] += 2.5 * h
            x[1] += 2.5 * t
            Xc.append(x)
            z = np.maximum(rng.normal(-1.0, 1.0, size=NF), 0)
            z[INTENT_FEATS] = np.maximum(rng.normal(2.5 * h - 0.5, 1.0, size=5), 0)
            z[LEX_FEATS] = np.maximum(rng.normal(2.5 * t - 0.5, 1.0, size=5), 0)
            Zc.append(z)
        for item, h in (("q_harm", 1), ("q_border", 0)):
            pair_rows.append({"group_id": f"g{gi}", "item": item, "split": split, "domain": DOMAINS[gi % 4],
                              "trigger": trig, "n_tokens": 20, "trigger_aligned": True})
            x = rng.normal(size=D)
            x[0] += 2.5 * h
            Xp.append(x)
    (root / "index").mkdir()
    ldir = root / "resid_post" / f"L{L:02d}"
    ldir.mkdir(parents=True)
    for name, rows in (("cells2x2", cell_rows), ("pairs", pair_rows)):
        (root / "index" / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    np.save(ldir / f"cells2x2__{POS}.npy", np.array(Xc, dtype=np.float16))
    np.save(ldir / f"pairs__{POS}.npy", np.array(Xp, dtype=np.float16))
    sp.save_npz(ldir / f"cells2x2__{POS}__16k.npz", sp.csr_matrix(np.array(Zc, dtype=np.float32)))
    return root


def _auc(block, key="gatilho_inedito"):
    return block["test"][key]["auroc"]


def test_cache_io_reads_contract(cache):
    assert cache_io.available(cache, "cells2x2") == [(L, POS)]
    rows = cache_io.load_index(cache, "cells2x2")
    assert len(rows) == 960 and rows[0]["cell"] == "harm_trig"
    assert cache_io.load_latents(cache, "cells2x2", L, POS, "16k").shape == (960, NF)


def test_load_latents_accepts_hand_made_csr(tmp_path):
    M = sp.csr_matrix(np.array([[0, 1.5, 0], [2.0, 0, 0]], dtype=np.float32))
    ldir = tmp_path / "resid_post" / "L03"
    ldir.mkdir(parents=True)
    np.savez(ldir / "pairs__post__16k.npz", indices=M.indices, values=M.data, indptr=M.indptr, d_sae=3)
    got = cache_io.load_latents(tmp_path, "pairs", 3, "post", "16k")
    assert (got.toarray() == M.toarray()).all()
    np.savez(ldir / "pairs__post__262k.npz", indices=M.indices, values=M.data, indptr=M.indptr)
    with pytest.raises(ValueError, match="shape"):
        cache_io.load_latents(tmp_path, "pairs", 3, "post", "262k")


def test_probes_recover_planted_directions(cache):
    res = probes.run(cache, n_boot=50)["resultados"]
    c = res["cells2x2"]["principal"][f"L{L}/{POS}"]
    assert _auc(c["cruzado_trig_para_notrig"]) > 0.9          # intenção sobrevive sem gatilho
    assert _auc(c["gatilho_independe_intencao"]) > 0.9        # gatilho independe da intenção
    assert abs(c["direcoes"]["cos_intencao_gatilho"]) < 0.3   # direções quase ortogonais
    assert c["direcoes"]["cos_intencao_recusa"] > 0.7         # intenção ≈ direção de recusa dos pares
    assert res["pairs"][f"L{L}/{POS}"]["test"]["todos"]["auroc"] > 0.9
    assert c["intencao"]["test"]["gatilho_inedito"]["n_groups"] > 0


def test_c2_double_dissociation(cache):
    rows = [r for r in cache_io.load_index(cache, "cells2x2")]
    Z = cache_io.load_latents(cache, "cells2x2", L, POS, "16k").toarray()
    res = classifiers.c2(Z, None, rows, INTENT_FEATS, LEX_FEATS, texts=None, n_boot=50, n_random=3)
    si, sl = res["matriz"]["S_intent"]["l1"], res["matriz"]["S_lex"]["l1"]
    assert classifiers._primary(si, "intencao_sem_gatilho") > 0.9
    assert classifiers._primary(si, "gatilho_todas") < 0.65
    assert classifiers._primary(sl, "gatilho_todas") > 0.9
    assert classifiers._primary(sl, "intencao_com_gatilho") < 0.65
    assert res["C2a"]["atende"] and res["C2c_consistencia"]["ok"]
    assert res["matriz"]["S_intent"]["media"]["intencao_com_gatilho"]["gatilho_inedito"]["auroc"] > 0.9
    assert "intencao_com_gatilho" in res["matriz"]["aleatorio"]
    assert res["C2b"]["status"].startswith("pendente")      # sem textos, o relativo fica pendente


def test_relative_drop_rule():
    rng = np.random.default_rng(1)
    n = 200
    y = np.array([1, 0] * n)
    g = np.repeat(np.arange(n), 2).astype(str)
    strong = y + rng.normal(0, 0.3, len(y))
    weak = y + rng.normal(0, 1.5, len(y))

    def fold(fi, fo, ti, to):
        return {"feat_in": fi, "feat_out": fo, "tf_in": ti, "tf_out": to, "y_in": y, "y_out": y, "g_in": g, "g_out": g}

    # features quase não caem; TF-IDF cai muito -> atende
    r = classifiers.relative_drop_folds([fold(strong, strong + rng.normal(0, 0.05, len(y)), strong, weak)], n_boot=200)
    assert r["atende"] is True and r["razao"] < 0.5
    # features caem tanto quanto o TF-IDF -> não atende
    r = classifiers.relative_drop_folds([fold(strong, weak, strong, weak)], n_boot=200)
    assert r["atende"] is False
    # TF-IDF não cai -> indefinido
    r = classifiers.relative_drop_folds([fold(strong, weak, strong, strong)], n_boot=50)
    assert r["atende"] is None and r["status"].startswith("indefinido")
    # várias dobras: média das quedas
    r = classifiers.relative_drop_folds([fold(strong, strong, strong, weak)] * 3, n_boot=50)
    assert r["n_dobras"] == 3 and r["atende"] is True


def test_c2b_three_generalizations(cache):
    """TF-IDF decora palavras de cada domínio; as features de intenção não dependem do domínio."""
    rng = np.random.default_rng(2)
    rows = cache_io.load_index(cache, "cells2x2")
    Z = cache_io.load_latents(cache, "cells2x2", L, POS, "16k").toarray()
    vocab = {d: ("".join(rng.choice(list("bcdfghjklmnpqrstvwxz"), 6)), "".join(rng.choice(list("bcdfghjklmnpqrstvwxz"), 6)))
             for d in DOMAINS}
    texts = {(r["group_id"], r["cell"]): f"please {'kill' if r['cell'].endswith('_trig') else 'stop'} the "
             f"{vocab[r['domain']][0] if r['cell'].startswith('harm') else vocab[r['domain']][1]} now" for r in rows}
    # XSTest sintético: mesmas features de intenção, vocabulário novo
    n = 120
    lab = np.array([1, 0] * (n // 2))
    Zx = np.maximum(rng.normal(-1.0, 1.0, size=(n, NF)), 0)
    Zx[:, INTENT_FEATS] = np.maximum(rng.normal(2.5 * lab[:, None] - 0.5, 1.0, size=(n, 5)), 0)
    xs_rows = [{"group_id": f"x{i}", "label": int(lab[i])} for i in range(n)]
    xs_texts = [f"could you {'qwvzk' if lab[i] else 'mrtpl'} it" for i in range(n)]

    res = classifiers.c2b(Z[:, INTENT_FEATS], rows, texts, n_boot=100, xstest=(Zx[:, INTENT_FEATS], xs_rows, xs_texts))
    assert res["leave_one_domain_out"]["n_dobras"] == 4
    assert res["leave_one_domain_out"]["atende"] is True      # TF-IDF cai no domínio novo; features não
    assert res["xstest"]["atende"] is True                    # idem no XSTest
    assert res["gatilho_inedito"]["atende"] is None           # aqui o TF-IDF não depende do gatilho
    assert res["n_atende"] == 2
    assert res["atende"] is res["xstest"]["atende"]          # só o XSTest decide (DESVIOS 2026-11-21)


def test_subsets_length_matched():
    rows = [{"group_id": "a", "n_tokens": n} for n in (20, 21, 22, 23)] + \
           [{"group_id": "b", "n_tokens": n} for n in (10, 20, 20, 20)]
    s = classifiers.subsets(rows, {"a": {"q_border_method": "retrieved"}, "b": {"q_border_method": "generated"}})
    assert s["c_tamanho_pareado"] == {"a"} and s["b_q0_retrieved"] == {"a"}
