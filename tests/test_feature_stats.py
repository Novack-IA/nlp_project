"""Testes de feature_stats.py (frente C) com um cache sintético no formato da seção 4 do AGENTS.md.

O cache tem features plantadas de cada tipo (intenção, lexical, mista, tema de um domínio, rara, nula).
A combinação (L22, inst_last) tem mais features de intenção, então deve ser a principal.
Os testes checam que cada feature cai na classe em que foi plantada, e só lá.
"""
import json

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from featgrpo.data.phase1.validate_template import CELLS
from featgrpo.interp import classifiers, feature_stats as fs

NF = 64
N_GROUPS = 800
DOMAINS = ("bias", "cyber", "privacy", "violence")
INTENT, LEX, MIXED, TOPIC, RARE = range(0, 6), range(6, 10), range(10, 12), range(12, 14), range(14, 16)
N_TOKEN_MISMATCH = 40          # grupos cujo último token do pedido muda entre as células


def _plant(rng, n_int):
    """(bI, bL, base, só_violência) por feature; as `n_int` primeiras de INTENT têm efeito de intenção."""
    spec = []
    for f in range(NF):
        if f in INTENT:
            spec.append((1.5 if f < n_int else 0.0, 0.0, -0.3, False))
        elif f in LEX:
            spec.append((0.0, 1.5, -0.3, False))
        elif f in MIXED:
            spec.append((1.2, 1.2, -0.3, False))
        elif f in TOPIC:
            spec.append((3.0, 0.0, -0.3, True))
        elif f in RARE:
            spec.append((1.5, 0.0, -4.0, False))
        else:
            spec.append((0.0, 0.0, -0.3, False))
    return spec


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("acts")
    rng = np.random.default_rng(1)
    train_trigs = [f"trig{i}" for i in range(40)]
    rows, meta = [], []
    for gi in range(N_GROUPS):
        split = "train" if gi < 0.70 * N_GROUPS else "val" if gi < 0.85 * N_GROUPS else "test"
        if split == "test" and rng.random() < 0.7:
            trig = f"novo{gi}"
        else:
            trig = train_trigs[rng.integers(0, 40)]
        dom = DOMAINS[gi % 4]
        mismatch = gi < N_TOKEN_MISMATCH
        meta.append((dom, mismatch))
        for ci, cell in enumerate(CELLS):
            rows.append({"group_id": f"g{gi}", "item": cell, "split": split, "domain": dom, "trigger": trig,
                         "n_tokens": 20, "trigger_aligned": True,
                         "inst_last_token_id": 100 + ci if mismatch else 236881})
    (root / "index").mkdir()
    (root / "index" / "cells2x2.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    # (L22, inst_last) tem as 6 features de intenção; as outras combinações têm só 2
    for layer in (9, 22):
        ldir = root / "resid_post" / f"L{layer:02d}"
        ldir.mkdir(parents=True)
        for pos in fs.MAIN_POSITIONS:
            n_int = 6 if (layer, pos) == (22, "inst_last") else 2
            spec = _plant(rng, n_int)
            Z = np.zeros((N_GROUPS * 4, NF), dtype=np.float32)
            for f, (b_i, b_l, base, so_viol) in enumerate(spec):
                b = rng.normal(base, 1.0, N_GROUPS)
                on = np.array([(d == "violence") if so_viol else True for d, _ in meta], dtype=float)
                for ci, cell in enumerate(CELLS):
                    h, t = int(cell.startswith("harm")), int(cell.endswith("_trig"))
                    z = b + b_i * h * on + b_l * t + rng.normal(0, 0.7, N_GROUPS)
                    Z[ci::4, f] = np.maximum(z, 0)
            sp.save_npz(ldir / f"cells2x2__{pos}__16k.npz", sp.csr_matrix(Z))

    # controles trig_only / neutral_only na combinação principal: lexicais e mistas reagem ao gatilho solto,
    # as de intenção não
    # formato real da frente A: 3 frases-molde por grupo do 2×2, `kind` = trig/neutral
    n_g, n_t = 100, 3
    nc = n_g * n_t
    crows = [{"group_id": f"g{gi}", "template_id": t, "kind": kd, "item_id": f"g{gi}:ctl{t}:{kd}",
              "inst_last_token_id": 236761}
             for gi in range(n_g) for t in range(n_t) for kd in ("trig", "neutral")]
    (root / "index" / "controls.jsonl").write_text("\n".join(json.dumps(r) for r in crows), encoding="utf-8")
    zc = np.zeros((nc * 2, NF), dtype=np.float32)
    for f in range(NF):
        b = np.repeat(rng.normal(-0.3, 1.0, n_g), n_t)          # frases do mesmo grupo compartilham a base
        for k, t in ((0, 1), (1, 0)):
            shift = 1.5 * t if (f in LEX or f in MIXED) else 0.0
            zc[k::2, f] = np.maximum(b + shift + rng.normal(0, 0.7, nc), 0)
    sp.save_npz(root / "resid_post" / "L22" / "controls__inst_last__16k.npz", sp.csr_matrix(zc))
    return root


@pytest.fixture(scope="module")
def run_desc(cache, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    res = fs.run_2x2(cache, "16k", (9, 22), fs.MAIN_POSITIONS, out, abrir_confirmacao=False, regras=fs.Regras())
    return res, out


# ---------------------------------------------------------------- peças

def test_efeitos_formulas():
    a = np.array([4.0, 2.0, 1.0, 1.0]).reshape(4, 1, 1)     # harm_trig, harm_notrig, ben_trig, ben_notrig
    i_g, l_g, x_g = fs.efeitos(a)
    assert (i_g.item(), l_g.item(), x_g.item()) == (2.0, 1.0, 1.0)


def test_cohen_d_e_ic():
    rng = np.random.default_rng(0)
    e = rng.normal(0.5, 1.0, (400, 3))
    d = fs.cohen_d(e)
    lo, hi = fs.ic_d(e, fs.pesos_bootstrap(400, 1000, 0))
    assert np.all((lo < d) & (d < hi))
    assert np.all((hi - lo > 0.15) & (hi - lo < 0.30))       # ~2 × 1,96 × √((1 + d²/2)/n) ≈ 0,21
    assert np.allclose(d, e.mean(0) / e.std(0, ddof=1))
    assert fs.cohen_d(np.zeros((10, 1))).item() == 0.0       # desvio zero não vira NaN


def test_cobertura_do_ic():
    """Em 200 amostras novas com d verdadeiro 0,5, o IC 95% contém 0,5 em ~95% das vezes."""
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 1.0, (400, 200))                       # 200 amostras como 200 colunas
    lo, hi = fs.ic_d(x, fs.pesos_bootstrap(400, 1000, 0))
    assert 0.90 <= np.mean((lo < 0.5) & (0.5 < hi)) <= 0.99


def test_p_wilcoxon_igual_ao_scipy_e_coluna_zero():
    rng = np.random.default_rng(0)
    e = np.maximum(rng.normal(0, 1, (300, 4)), 0) - np.maximum(rng.normal(0, 1, (300, 4)), 0)
    e[:, 3] = 0
    p = fs.p_wilcoxon(e)
    ref = wilcoxon(e[:, 0], zero_method="pratt", method="approx").pvalue
    assert np.isclose(p[0], ref) and p[3] == 1.0


def test_bh_igual_statsmodels():
    p = np.random.default_rng(0).uniform(0, 1, 500) ** 3
    assert np.allclose(fs.bh(p), multipletests(p, method="fdr_bh")[1])


# ---------------------------------------------------------------- pipeline (descoberta)

def _tabela(out, layer, pos):
    import pandas as pd
    return pd.read_csv(out / f"feature_classes_L{layer}_{pos}_16k.csv")


def test_combinacao_principal_e_confirmacao_fechada(run_desc):
    res, out = run_desc
    assert res["principal"]["camada"] == 22 and res["principal"]["posicao"] == "inst_last"
    assert res["C1"]["status"] == "aguardando_confirmacao"
    assert res["replicacao_S_intent"] is None
    assert not any(c.startswith("d_I_test") for c in _tabela(out, 22, "inst_last").columns)


def test_classes_recuperam_o_que_foi_plantado(run_desc):
    _, out = run_desc
    df = _tabela(out, 22, "inst_last").set_index("feature")
    cls = df["classe"]
    assert set(INTENT) <= set(cls[cls == "S_intent"].index)
    assert set(LEX) <= set(cls[cls == "S_lex"].index)
    assert set(MIXED) <= set(cls[cls == "S_mixed"].index)
    nulas = [f for f in range(16, NF) if f in df.index]
    assert not cls.loc[nulas].isin(["S_intent", "S_lex", "S_mixed"]).any()
    assert not set(RARE) & set(df.index)          # raras não passam do filtro de 5% (nem entram no CSV)


def test_tema_de_um_dominio_nao_e_geral(run_desc):
    res, _ = run_desc
    por = {f["feature"]: f for f in res["por_dominio"]["features"]}
    assert all(por[f]["geral"] for f in INTENT)
    for f in TOPIC:
        if f in por:                                   # se passou como S_intent, é marcada como tema
            assert not por[f]["geral"] and por[f]["dominios_consistentes"] <= 1


def test_exclusao_do_token_no_fim_pedido(run_desc):
    res, _ = run_desc
    ex = res["grupos_excluidos"]
    assert ex["L22_inst_last_16k"]["token_diferente_fim_pedido"] == N_TOKEN_MISMATCH
    assert ex["L22_inst_last_16k"]["fim_pedido_token_verificado"] is True
    assert "token_diferente_fim_pedido" not in ex["L22_post_last_16k"]


def test_listas_no_formato_da_frente_d(run_desc):
    _, out = run_desc
    feats = classifiers.load_list(out / "listas" / "S_intent_L22_inst_last_16k.json")
    assert set(INTENT) <= set(feats)
    payload = json.loads((out / "listas" / "S_intent_L22_inst_last_16k.json").read_text())
    assert payload["principal"] is True and payload["rotulo"] == "fim_pedido"


# ---------------------------------------------------------------- confirmação, pisos, C4a

def test_confirmacao_replica(cache, tmp_path):
    res = fs.run_2x2(cache, "16k", (9, 22), fs.MAIN_POSITIONS, tmp_path, abrir_confirmacao=True, regras=fs.Regras())
    rep = res["replicacao_S_intent"]
    assert rep["recorte_usado"] == "inedito" and not rep["baixo_poder"]
    assert rep["fracao_replica"] >= 0.7 and res["C1"]["status"] == "atende"
    assert "d_I_test_inedito" in _tabela(tmp_path, 22, "inst_last").columns


def test_piso_de_confirmacao_usa_test_inteiro(cache, tmp_path):
    res = fs.run_2x2(cache, "16k", (22,), ("inst_last",), tmp_path, abrir_confirmacao=True,
                     regras=fs.Regras(piso_confirmacao=10_000))
    assert res["replicacao_S_intent"]["recorte_usado"] == "todos" and res["replicacao_S_intent"]["baixo_poder"]


def test_piso_de_descoberta_da_inconclusivo(cache, tmp_path):
    res = fs.run_2x2(cache, "16k", (22,), ("inst_last",), tmp_path, abrir_confirmacao=False,
                     regras=fs.Regras(piso_descoberta=10_000))
    assert res["principal"] is None and res["C1"]["status"] == "inconclusivo"


def test_c4a_usa_a_combinacao_forcada(cache, tmp_path):
    res = fs.run_2x2(cache, "16k", (9, 22), fs.MAIN_POSITIONS, tmp_path, abrir_confirmacao=False,
                     regras=fs.Regras(), principal_forcada=(9, "post_last"))
    assert res["principal"] == {"camada": 9, "posicao": "post_last", "rotulo": "pre_resposta", "forcada_pelo_16k": True}


def test_modo_provisorio_nos_pares(tmp_path):
    """Pares originais: a feature 0 acende mais em Q⁺, a 1 mais em Q⁰; o resto é nulo."""
    rng = np.random.default_rng(2)
    n = 600
    rows = []
    for gi in range(n):
        split = "train" if gi < 420 else "val" if gi < 510 else "test"
        for item in ("q_harm", "q_border"):
            rows.append({"group_id": f"p{gi}", "item": item, "split": split, "domain": DOMAINS[gi % 4],
                         "trigger": "kill", "inst_last_token_id": 236881})
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "pairs.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ldir = tmp_path / "resid_post" / "L22"
    ldir.mkdir(parents=True)
    Z = np.zeros((2 * n, 8), dtype=np.float32)
    for f in range(8):
        b = rng.normal(-0.3, 1.0, n)
        for k, h in ((0, 1), (1, 0)):
            shift = 1.5 * h if f == 0 else 1.5 * (1 - h) if f == 1 else 0.0
            Z[k::2, f] = np.maximum(b + shift + rng.normal(0, 0.7, n), 0)
    sp.save_npz(ldir / "pairs__inst_last__16k.npz", sp.csr_matrix(Z))
    res = fs.run_provisorio(tmp_path, "16k", (22,), ("inst_last",), tmp_path / "out", fs.Regras())
    c = res["combinacoes"][0]
    assert c["top20_mais_ativas_em_Q_harm"] == [0] and c["top20_mais_ativas_em_Q_border"] == [1]
    assert c["n_grupos_descoberta"] == 510 and "diagnóstico" in res["aviso"]


def test_regra_antiga_e_mais_frouxa(run_desc):
    res, _ = run_desc
    for c in res["combinacoes_descoberta"]:
        assert c["S_intent_regra_antiga"] >= c["S_intent"]


# ---------------------------------------------------------------- C3: controle, fichas cegas e figura

def test_controle_trig_only(run_desc):
    c = run_desc[0]["C3_controle_trig_only"]
    assert c["status"] == "ok" and c["n_pares_controle"] == 300
    assert c["S_intent"]["atende"] and c["S_intent"]["fracao_sem_efeito"] > 0.5
    assert c["S_lex"]["fracao_que_reage"] == 1.0


def test_controle_unidade_grupo(cache, run_desc):
    from featgrpo.interp import cache_io
    _, out = run_desc
    import pandas as pd
    df = pd.read_csv(out / "feature_classes_L22_inst_last_16k.csv")
    z = cache_io.load_latents(cache, "controls", 22, "inst_last", "16k")
    rows = cache_io.load_index(cache, "controls")
    si, sl = (df.loc[df["classe"] == c, "feature"].to_numpy() for c in ("S_intent", "S_lex"))
    frase = fs.controle_trig_only(z, rows, "inst_last", si, sl, fs.Regras(), unidade="frase")
    grupo = fs.controle_trig_only(z, rows, "inst_last", si, sl, fs.Regras(), unidade="grupo")
    assert frase["n_pares_controle"] == 300 and grupo["n_unidades"] == 100
    assert grupo["S_lex"]["fracao_que_reage"] == 1.0
    with pytest.raises(ValueError):
        fs.controle_trig_only(z, rows, "inst_last", si, sl, fs.Regras(), unidade="outra")


def test_pesos_bootstrap_por_grupo():
    grupo = np.array(["a", "a", "b", "c", "c", "c"], dtype=object)
    w = fs.pesos_bootstrap_por_grupo(grupo, 200, 0)
    assert w.shape == (200, 6)
    assert np.array_equal(w[:, 0], w[:, 1]) and np.array_equal(w[:, 3], w[:, 5])
    # cada reamostra tem 3 grupos; o total de frases varia com os grupos sorteados
    tam = np.array([2, 1, 3])
    gw = np.stack([w[:, 0], w[:, 2], w[:, 3]], axis=1)
    assert np.all(gw.sum(axis=1) == 3) and np.allclose(w.sum(axis=1), gw @ tam)


def test_ic_d_igual_ao_bootstrap_simples_antigo():
    rng = np.random.default_rng(1)
    e = rng.normal(0.3, 1.0, (120, 5))
    w = fs.pesos_bootstrap(120, 300, 0)
    lo, hi = fs.ic_d(e, w)
    n = 120   # fórmula de antes da generalização (total de cada reamostra = n)
    m = w @ e / n
    s = np.sqrt((w @ (e * e) / n - m * m) * (n / (n - 1)))
    lo0, hi0 = np.percentile(m / s, [2.5, 97.5], axis=0)
    assert np.allclose(lo, lo0) and np.allclose(hi, hi0)


def test_reamostragem_por_grupo_alarga_o_ic_com_frases_dependentes():
    rng = np.random.default_rng(2)
    n_g, n_t = 150, 3
    e = np.repeat(rng.normal(0.0, 1.0, (n_g, 4)), n_t, axis=0) + rng.normal(0, 0.2, (n_g * n_t, 4))
    grupo = np.repeat(np.arange(n_g), n_t)
    lo_f, hi_f = fs.ic_d(e, fs.pesos_bootstrap(len(e), 1000, 0))
    lo_g, hi_g = fs.ic_d(e, fs.pesos_bootstrap_por_grupo(grupo, 1000, 0))
    assert np.all((hi_g - lo_g) > 1.4 * (hi_f - lo_f))     # ≈ √3 quando as 3 frases são quase iguais


def test_controle_principal_e_sensibilidade(run_desc):
    res = run_desc[0]
    c, s = res["C3_controle_trig_only"], res["C3_controle_trig_only_sensibilidade_reamostragem_frase"]
    assert c["reamostragem"] == "grupo" and c["n_grupos_2x2"] == 100
    assert s["reamostragem"] == "frase" and "features" not in s
    assert s["S_intent"]["n"] == c["S_intent"]["n"]
    larg_g = np.mean([f["ic_hi"] - f["ic_lo"] for f in c["features"]])
    assert larg_g > 0


def test_controle_ausente_fica_registrado(cache, tmp_path):
    res = fs.run_2x2(cache, "16k", (22,), ("inst_last",), tmp_path, abrir_confirmacao=False, regras=fs.Regras(),
                     conjunto_controles="nao_existe")
    assert res["C3_controle_trig_only"]["status"].startswith("sem cache")


def _fake_np(layer, sae, index):
    return {"explanations": [{"description": f"conceito {index}"}],
            "activations": [{"tokens": ["<bos>", "How", " to", " kill", "?"], "values": [0, 0, 1, 9, 2]}],
            "pos_str": ["kill", "stop"]}


def test_fichas_cegas(cache, run_desc):
    from featgrpo.interp import autointerp, cache_io
    res, _ = run_desc
    rows = cache_io.load_index(cache, "cells2x2")
    z = cache_io.load_latents(cache, "cells2x2", 22, "inst_last", "16k")
    textos = {(r["group_id"], r["cell"]): f"texto {r['group_id']} {r['cell']}" for r in rows}
    fichas, chave = autointerp.montar_fichas(res, "16k", z, rows, textos, fetch=_fake_np)
    n = len(res["top20"]["S_intent"]) + len(res["top20"]["S_lex"])
    assert len(fichas) == len(chave) == n
    # cegas: nada nas fichas revela a classe ou o d
    assert not {"classe", "feature", "d_I", "d_L"} & set(fichas.columns)
    assert not fichas.astype(str).apply(lambda c: c.str.contains("S_intent|S_lex")).any().any()
    # ordem aleatória, mas reprodutível
    _, chave2 = autointerp.montar_fichas(res, "16k", z, rows, textos, fetch=_fake_np)
    assert chave["feature"].tolist() == chave2["feature"].tolist()
    assert chave["classe"].tolist() != sorted(chave["classe"].tolist(), reverse=True)
    # os itens do 2×2 vêm só da descoberta
    test_groups = {r["group_id"] for r in rows if r["split"] == "test"}
    itens = "\n".join(fichas["itens_2x2_com_maior_ativacao"])
    assert itens and not any(f"texto {g} " in itens for g in test_groups)
    assert "[[ kill]]" in fichas["exemplos_neuronpedia"].iloc[0]


def test_avaliacao_cega():
    import pandas as pd
    from featgrpo.interp import autointerp
    chave = pd.DataFrame({"ficha": [f"F{i}" for i in range(6)], "classe": ["S_intent"] * 4 + ["S_lex"] * 2})
    fred = pd.DataFrame({"ficha": chave["ficha"], "rotulo": ["conceitual"] * 3 + ["lexical"] * 3})
    davi = pd.DataFrame({"ficha": chave["ficha"], "rotulo": ["conceitual", "conceitual", "lexical", "lexical", "lexical", "lexical"]})
    r = autointerp.avaliar(fred, davi, chave)
    assert r["fracao_S_intent_conceitual_pelos_dois"] == 0.5 and r["atende"] is False
    assert r["concordancia_bruta"] == 5 / 6 and 0 < r["kappa_cohen"] < 1
    with pytest.raises(ValueError):
        autointerp.avaliar(fred.assign(rotulo="talvez"), davi, chave)


def test_avaliacao_varios_avaliadores():
    import pandas as pd
    from featgrpo.interp import autointerp
    chave = pd.DataFrame({"ficha": [f"F{i}" for i in range(5)], "classe": ["S_intent"] * 5})
    llm = pd.DataFrame({"ficha": chave["ficha"], "rotulo": ["conceitual"] * 4 + ["lexical"]})
    fred = pd.DataFrame({"ficha": chave["ficha"], "rotulo": ["conceitual"] * 3 + ["lexical"] * 2})
    r = autointerp.avaliar_varios({"llm": llm}, chave, principal="llm")
    assert r["fracao_S_intent_conceitual"]["llm"] == 0.8 and r["atende"] is True and r["pares"] == {}
    r2 = autointerp.avaliar_varios({"llm": llm, "frederico": fred}, chave, principal="llm")
    assert r2["fracao_S_intent_conceitual_para_todos"] == 0.6 and r2["pares"]["llm×frederico"]["concordancia_bruta"] == 0.8


def test_fichas_sem_nome_da_celula(cache, run_desc):
    from featgrpo.interp import autointerp, cache_io
    rows = cache_io.load_index(cache, "cells2x2")
    z = cache_io.load_latents(cache, "cells2x2", 22, "inst_last", "16k")
    textos = {(r["group_id"], r["cell"]): "um pedido" for r in rows}
    itens = autointerp.top_itens_2x2(z, rows, [0, 1], textos)
    assert itens[0] and not any(c in s for s in itens[0] for c in ("harm_", "ben_"))
    assert any("harm_" in s or "ben_" in s for s in autointerp.top_itens_2x2(z, rows, [0], textos, mostrar_celula=True)[0])


def test_lista_exploratoria(run_desc):
    from featgrpo.interp import autointerp, classifiers
    res, out = run_desc
    fake = lambda layer, sae, f: {"explanations": [{"description": "cannot fulfill requests" if f % 2 else "x"}]}
    lst = autointerp.lista_exploratoria({"16k": res}, out, fetch=fake)
    assert lst["exploratorio"] is True and lst["classe"] == "intencao_forte_fora_de_S_intent"
    import pandas as pd
    df = pd.read_csv(out / "feature_classes_L22_inst_last_16k.csv").set_index("feature")
    assert all(df.loc[f, "classe"] != "S_intent" and abs(df.loc[f, "d_I"]) >= 0.5 for f in lst["features"])
    assert lst["n_rotulo_de_recusa"] == sum(f % 2 for f in lst["features"])
    p = out / "lst.json"; p.write_text(__import__("json").dumps(lst))
    assert classifiers.load_list(p) == lst["features"]


def test_por_tipo_gatilho(cache, tmp_path):
    tipos = {f"g{i}": fs.TIPOS_GATILHO[i % 3] for i in range(N_GROUPS)}
    res = fs.run_2x2(cache, "16k", (22,), ("inst_last",), tmp_path, abrir_confirmacao=False, regras=fs.Regras(),
                     tipos_gatilho=tipos)
    pt = res["por_tipo_gatilho"]
    assert pt["descritivo"] is True and set(pt["tipos"]) == set(fs.TIPOS_GATILHO)
    # as features de intenção plantadas valem em todos os tipos
    for t_ in fs.TIPOS_GATILHO:
        assert pt["tipos"][t_]["S_intent_com_mesmo_sinal_e_d_ge_0_3"] >= len(INTENT)
    assert "por_tipo_gatilho" not in fs.run_2x2(cache, "16k", (22,), ("inst_last",), tmp_path / "b",
                                                abrir_confirmacao=False, regras=fs.Regras())


def test_figura_di_dl(run_desc, tmp_path):
    import pandas as pd
    from featgrpo.interp import figuras
    _, out = run_desc
    tabelas = {(L, p): pd.read_csv(out / f"feature_classes_L{L}_{p}_16k.csv") for L in (9, 22) for p in fs.MAIN_POSITIONS}
    png = figuras.scatter_di_dl(tabelas, tmp_path / "fig.png", "16k", principal=(22, "inst_last"))
    assert png.stat().st_size > 20_000 and png.with_suffix(".pdf").exists()


# ---------------------------------------------------------------- caminho (c): controle de estilo

def test_controle_estilo_separa_estilo_de_intencao():
    rng = np.random.default_rng(3)
    n = 800
    ds = np.column_stack([rng.normal(1.0, 1.0, n), rng.normal(0.5, 1.0, n)])   # o nocivo tende a ser "mais longo"
    i_estilo = 1.2 * ds[:, 0] + rng.normal(0, 0.5, n)        # efeito que vem só da diferença de estilo
    i_intencao = 0.8 + rng.normal(0, 1.0, n)                   # efeito que existe com estilo igual
    i = np.column_stack([i_estilo, i_intencao])
    d = fs.cohen_d(i)
    assert np.all(np.abs(d) >= 0.5)                            # as duas passariam como S_intent pelo d
    resumo, df = fs.controle_estilo(i, ds, d, fs.Regras())
    assert df["sobrevive_estilo"].tolist() == [False, True]
    assert df["r2_estilo"].iloc[0] > 0.5 and df["r2_estilo"].iloc[1] < 0.05
    assert resumo["n_sobrevivem"] == 1 and resumo["atributos_de_estilo_usados"] == 2


def test_controle_estilo_no_fluxo_2x2(cache, tmp_path):
    from featgrpo.interp import cache_io
    rows = cache_io.load_index(cache, "cells2x2")
    rng = np.random.default_rng(4)
    textos = {(r["group_id"], r["cell"]): "word " * int(rng.integers(5, 30)) + ("?" if rng.random() < 0.8 else ".")
              for r in rows}
    res = fs.run_2x2(cache, "16k", (22,), ("inst_last",), tmp_path, abrir_confirmacao=False, regras=fs.Regras(),
                     textos=textos)
    e = res["C1_controle_estilo"]
    assert e["status"] == "ok" and e["n_features"] == res["combinacoes_descoberta"][0]["S_intent"]
    assert e["fracao_sobrevive"] >= 0.8                        # as plantadas são de intenção, não de estilo
    sem = fs.run_2x2(cache, "16k", (22,), ("inst_last",), tmp_path, abrir_confirmacao=False, regras=fs.Regras(),
                     textos={})
    assert sem["C1_controle_estilo"]["status"].startswith("sem textos")


# ---------------------------------------------------------------- subconjunto do QC (DESVIOS 2026-11-21)

def test_subconjunto_filtra_grupos_e_controles(cache, tmp_path):
    permitidos = {f"g{i}" for i in range(50, N_GROUPS)}             # tira os 50 primeiros
    res = fs.run_2x2(cache, "16k", (22,), ("inst_last",), tmp_path, abrir_confirmacao=False, regras=fs.Regras(),
                     grupos_permitidos=permitidos, subconjunto="qc_grammatical")
    ex = res["grupos_excluidos"]["L22_inst_last_16k"]
    assert ex["fora_do_subconjunto"] == 50 - N_TOKEN_MISMATCH        # os 40 de token diferente já tinham saído
    assert res["subconjunto"] == "qc_grammatical"
    assert res["C3_controle_trig_only"]["n_pares_controle"] == 150   # g50…g99, 3 frases cada


def test_subconjunto_pequeno_fica_inconclusivo(cache, tmp_path):
    permitidos = {f"g{i}" for i in range(N_TOKEN_MISMATCH, N_TOKEN_MISMATCH + 200)}
    res = fs.run_2x2(cache, "16k", (22,), ("inst_last",), tmp_path, abrir_confirmacao=False, regras=fs.Regras(),
                     grupos_permitidos=permitidos, principal_forcada=(22, "inst_last"), subconjunto="qc_pass")
    assert res["C1"]["status"] == "inconclusivo"                    # < 260 grupos na descoberta


def test_grupos_do_subconjunto(tmp_path):
    p = tmp_path / "cells.jsonl"
    p.write_text("\n".join(json.dumps({"group_id": f"g{i}", "qc_grammatical": i % 2 == 0, "qc_pass": i == 0,
                                       "validator_pass": i != 2})
                           for i in range(6)), encoding="utf-8")
    assert fs.grupos_do_subconjunto("qc_grammatical", p) == {"g0", "g4"}                  # g2 reprovado
    assert fs.grupos_do_subconjunto("qc_grammatical", p, validado=False) == {"g0", "g2", "g4"}
    assert fs.grupos_do_subconjunto("qc_pass", p) == {"g0"}
    assert fs.grupos_do_subconjunto("todos", p) is None
    with pytest.raises(ValueError):
        fs.grupos_do_subconjunto("outro", p)
    sem_marca = tmp_path / "antigo.jsonl"
    sem_marca.write_text(json.dumps({"group_id": "g0", "qc_grammatical": True}), encoding="utf-8")
    with pytest.raises(SystemExit):
        fs.grupos_do_subconjunto("qc_grammatical", sem_marca)
    assert fs.grupos_do_subconjunto("qc_grammatical", sem_marca, validado=False) == {"g0"}


def test_diagnostico_lexical(tmp_path):
    """Intenção numa direção fixa; gatilho numa direção diferente por grupo, mais um empurrão pequeno no eixo da
    intenção. O diagnóstico deve achar a intenção consistente, o gatilho inconsistente e o empurrão positivo."""
    rng = np.random.default_rng(3)
    n, d = 300, 32
    rows = [{"group_id": f"g{i}", "item": c, "split": "train", "domain": "bias", "trigger": f"t{i}",
             "inst_last_token_id": 7} for i in range(n) for c in CELLS]
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "cells2x2.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    eixo = np.zeros(d); eixo[0] = 1.0
    x = np.zeros((n, 4, d))
    for i in range(n):
        base, lex = rng.normal(0, 1, d), rng.normal(0, 1, d) * 2.0 + 0.1 * eixo * 5
        for ci, c in enumerate(CELLS):
            h, t = int(c.startswith("harm")), int(c.endswith("_trig"))
            x[i, ci] = base + 5 * eixo * h + lex * t + rng.normal(0, 0.1, d)
    ldir = tmp_path / "resid_post" / "L22"; ldir.mkdir(parents=True)
    np.save(ldir / "cells2x2__inst_last.npy", x.reshape(n * 4, d).astype(np.float32))
    sp.save_npz(ldir / "cells2x2__inst_last__16k.npz", sp.csr_matrix(np.maximum(x.reshape(n * 4, d), 0).astype(np.float32)))
    (r,) = fs.diagnostico_lexical(tmp_path, "16k", (22,), ("inst_last",))
    assert r["n_grupos_descoberta"] == n
    assert r["consistencia_I"] > 0.9 and r["consistencia_L"] < 0.3
    assert r["cos_L_medio_I_medio"] > 0 and 0 < r["L_no_eixo_da_intencao_razao"] < 0.3
    assert fs.consistencia(np.ones((5, 3))) == pytest.approx(1.0)


def test_descoberta_md(run_desc):
    from featgrpo.interp import autointerp
    res, out = run_desc
    fake = lambda layer, sae, f: {"explanations": [{"description": f"conceito {f}"}]}
    md = autointerp.descoberta_md({"16k": res}, out, fetch=fake)
    assert "# C1 no 2×2: o que a descoberta mostra" in md and "## Em resumo" in md
    import re
    assert re.search(r"\| L22 \| \*\*\d+\*\* \(principal\) \|", md) and "## Onde está o sinal" in md
    assert "conceito " in md and "Só a descoberta" in md and "## Anexo" in md
    assert autointerp.RECUSA.search("cannot fulfill requests") and not autointerp.RECUSA.search("illegal activities")


def test_provisorio_md(tmp_path):
    import pandas as pd
    from featgrpo.interp import autointerp
    pd.DataFrame({"feature": [5, 9], "d_int": [0.8, -0.6], "freq_ativacao": [0.3, 0.1]}).to_csv(
        tmp_path / "provisorio_L22_inst_last_16k.csv", index=False)
    prov = {"git_commit": "abc1234def", "combinacoes": [{
        "camada": 22, "posicao": "inst_last", "rotulo": "fim_pedido", "n_grupos_descoberta": 100, "n_sig_d05": 2,
        "top20_mais_ativas_em_Q_harm": [5], "top20_mais_ativas_em_Q_border": [],
        "controle_estilo": {"status": "ok", "n_sobrevivem": 2, "n_features": 2, "r2_mediano": 0.1}}]}
    fake = lambda layer, sae, f: {"explanations": [{"description": "a | b\nc"}]}
    md = autointerp.provisorio_md(prov, tmp_path, "16k", fetch=fake)
    assert "[5](https://www.neuronpedia.org/gemma-3-4b-it/22-gemmascope-2-res-16k/5)" in md
    assert "| 0,80 |" in md and "Nenhuma feature com efeito grande" in md
    assert "a \\| b c" in md and "Diagnóstico, fora do veredito" in md and "2 de 2" in md
