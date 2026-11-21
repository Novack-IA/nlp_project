"""Fichas cegas do C3 (frente C), com as explicações do Neuronpedia.

Pré-registro, seção 5, C3 item 2: uma ficha por feature do top-20 de S_intent e do top-20 de S_lex da
combinação principal, com a explicação e os exemplos do Neuronpedia e os 10 itens do 2×2 em que a feature
mais acende, só da descoberta. Ordem aleatória (semente fixa), sem dizer a classe. Frederico e Davi rotulam
cada ficha como `conceitual` (intenção, dano, ação perigosa) ou `lexical` (palavra ou expressão específica).
O "autointerp" vem do Neuronpedia (seção 2): não há LLM local.

    PYTHONPATH=src python -m featgrpo.interp.autointerp montar --sae 16k
    PYTHONPATH=src python -m featgrpo.interp.autointerp avaliar --frederico f.csv --davi d.csv

Saídas de `montar`:
- `results/frente-c/fichas_cegas.csv`: o que os avaliadores veem, sem classe e sem d, com a coluna `rotulo`
  vazia. Cada avaliador preenche a própria cópia, sem ver a do outro.
- `artifacts/frente-c/fichas_chave.csv`: a chave (ficha → feature, classe, d). Fica fora do git até a
  avaliação acabar, para as fichas continuarem cegas.
- `artifacts/neuronpedia/...`: cache das respostas da API, para não buscar duas vezes.

Saída de `avaliar`: a seção C3 da avaliação cega em `results/frente-c/c1_c3.json` (fração de S_intent
conceituais para os dois avaliadores e o κ de Cohen entre eles).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from featgrpo.interp import cache_io
from featgrpo.interp.data import PHASE1, REPO

NP_API = "https://www.neuronpedia.org/api/feature/{model}/{source}/{index}"
NP_PAGE = "https://www.neuronpedia.org/{model}/{source}/{index}"
NP_MODEL = "gemma-3-4b-it"
OUT_DIR = REPO / "results" / "frente-c"
KEY_DIR = REPO / "artifacts" / "frente-c"
NP_CACHE = REPO / "artifacts" / "neuronpedia"
DISCOVERY = ("train", "val")
ROTULOS = ("conceitual", "lexical")


def np_source(layer: int, sae: str) -> str:
    return f"{layer}-gemmascope-2-res-{sae}"


def fetch_feature(layer: int, sae: str, index: int, cache_dir: Path = NP_CACHE) -> dict:
    """Ficha de uma feature na API pública do Neuronpedia, com cache em disco."""
    p = Path(cache_dir) / NP_MODEL / np_source(layer, sae) / f"{index}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    url = NP_API.format(model=NP_MODEL, source=np_source(layer, sae), index=index)
    req = urllib.request.Request(url, headers={"User-Agent": "research-client"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    time.sleep(0.25)                       # gentileza com a API pública
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data), encoding="utf-8")
    return data


def _limpa(s: str) -> str:
    return s.replace("\n", "⏎").replace("\r", "")


def resumo_neuronpedia(d: dict, n_ex: int = 5, antes: int = 12, depois: int = 4) -> dict:
    """Explicação, exemplos (token de maior ativação entre [[ ]]) e palavras promovidas de uma ficha."""
    expl = [e.get("description", "").strip() for e in d.get("explanations") or [] if e.get("description")]
    exemplos = []
    for a in (d.get("activations") or [])[:n_ex]:
        toks, vals = a.get("tokens") or [], a.get("values") or []
        if not toks:
            continue
        i = int(np.argmax(vals))
        ctx = "".join(toks[max(0, i - antes):i]) + "[[" + toks[i] + "]]" + "".join(toks[i + 1:i + 1 + depois])
        exemplos.append(_limpa(ctx))
    return {"explicacao": " | ".join(expl) or "(sem explicação no Neuronpedia)",
            "exemplos": exemplos, "promove": [_limpa(t) for t in (d.get("pos_str") or [])[:8]]}


def textos_2x2(path: Path = PHASE1 / "cells_2x2.jsonl") -> dict[tuple[str, str], str]:
    """(group_id, célula) → texto, a partir do arquivo da frente A."""
    out = {}
    if not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                g = json.loads(line)
                for c, v in (g.get("cells") or {}).items():
                    out[(g["group_id"], c)] = v.get("text", "")
    return out


def top_itens_2x2(z: sp.csr_matrix, rows: list[dict], feats: list[int], textos: dict, k: int = 10,
                  mostrar_celula: bool = False) -> dict[int, list[str]]:
    """Para cada feature, os k itens do 2×2 (só descoberta) com maior ativação. Sem o nome da célula por padrão,
    porque ele diria ao avaliador se o pedido é nocivo (DESVIOS 2026-11-21)."""
    keep = [i for i, r in enumerate(rows) if r.get("cell") and r.get("split") in DISCOVERY]
    sub = z[keep][:, feats].toarray()
    out = {}
    for j, f in enumerate(feats):
        order = [i for i in np.argsort(-sub[:, j])[:k] if sub[i, j] > 0]
        out[f] = [(f"{rows[keep[i]]['cell']}: " if mostrar_celula else "")
                  + f"{_limpa(textos.get((rows[keep[i]]['group_id'], rows[keep[i]]['cell']), '(texto não encontrado)'))}"
                  f" (ativação {sub[i, j]:.1f})" for i in order]
    return out


def montar_fichas(resumo_sae: dict, sae: str, z: sp.csr_matrix, rows: list[dict], textos: dict,
                  fetch: Callable[[int, str, int], dict] = fetch_feature, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fichas (o que os avaliadores veem) e chave (ficha → feature, classe, d), em ordem aleatória."""
    p = resumo_sae["principal"]
    layer, pos = int(p["camada"]), p["posicao"]
    feats = [(int(f["feature"]), cls, f.get("d_I"), f.get("d_L"))
             for cls in ("S_intent", "S_lex") for f in resumo_sae["top20"][cls]]
    tops = top_itens_2x2(z, rows, [f for f, *_ in feats], textos)
    order = np.random.default_rng(seed).permutation(len(feats))
    fichas, chave = [], []
    for k, idx in enumerate(order, 1):
        f, cls, d_i, d_l = feats[idx]
        npd = resumo_neuronpedia(fetch(layer, sae, f))
        ficha = f"F{k:02d}"
        fichas.append({"ficha": ficha, "explicacao_neuronpedia": npd["explicacao"],
                       "exemplos_neuronpedia": "\n".join(npd["exemplos"]),
                       "palavras_que_a_feature_promove": ", ".join(npd["promove"]),
                       "itens_2x2_com_maior_ativacao": "\n".join(tops.get(f, [])),
                       "rotulo": ""})
        chave.append({"ficha": ficha, "feature": f, "classe": cls, "d_I": d_i, "d_L": d_l, "camada": layer,
                      "posicao": pos, "sae": sae,
                      "neuronpedia": NP_PAGE.format(model=NP_MODEL, source=np_source(layer, sae), index=f)})
    return pd.DataFrame(fichas), pd.DataFrame(chave)


def avaliar_varios(avaliadores: dict[str, pd.DataFrame], chave: pd.DataFrame, principal: str,
                   limiar: float = 0.60) -> dict:
    """C3, item 2, com qualquer número de avaliadores (DESVIOS 2026-11-21: primeiro um LLM, depois a conferência
    humana). O critério usa o avaliador `principal`; reportamos também a fração conceitual para todos e o κ de
    Cohen de cada par."""
    from itertools import combinations
    from sklearn.metrics import cohen_kappa_score
    df = chave.copy()
    for nome, rot in avaliadores.items():
        df = df.merge(rot[["ficha", "rotulo"]].rename(columns={"rotulo": nome}), on="ficha")
        df[nome] = df[nome].astype(str).str.strip().str.lower()
        invalidos = sorted(set(df[nome]) - set(ROTULOS))
        if invalidos:
            raise ValueError(f"rótulos inválidos de {nome}: {invalidos} (use {ROTULOS})")
    si = df[df["classe"] == "S_intent"]
    por = {nome: float((si[nome] == "conceitual").mean()) if len(si) else None for nome in avaliadores}
    todos = float(si[list(avaliadores)].eq("conceitual").all(axis=1).mean()) if len(si) else None
    pares = {f"{a}×{b}": {"kappa_cohen": float(cohen_kappa_score(df[a], df[b])) if df[a].nunique() + df[b].nunique() > 2
                          else None, "concordancia_bruta": float((df[a] == df[b]).mean())}
             for a, b in combinations(avaliadores, 2)}
    frac = por[principal]
    return {"n_fichas": len(df), "n_S_intent": len(si), "avaliador_principal": principal,
            "fracao_S_intent_conceitual": por, "fracao_S_intent_conceitual_para_todos": todos,
            "atende": None if frac is None else frac >= limiar, "limiar": limiar, "pares": pares}


def avaliar(fred: pd.DataFrame, davi: pd.DataFrame, chave: pd.DataFrame, limiar: float = 0.60) -> dict:
    """C3, item 2: fração das fichas de S_intent rotuladas `conceitual` pelos dois, e κ de Cohen."""
    from sklearn.metrics import cohen_kappa_score
    df = chave.merge(fred[["ficha", "rotulo"]].rename(columns={"rotulo": "frederico"}), on="ficha") \
              .merge(davi[["ficha", "rotulo"]].rename(columns={"rotulo": "davi"}), on="ficha")
    for col in ("frederico", "davi"):
        df[col] = df[col].astype(str).str.strip().str.lower()
        invalidos = sorted(set(df[col]) - set(ROTULOS))
        if invalidos:
            raise ValueError(f"rótulos inválidos de {col}: {invalidos} (use {ROTULOS})")
    si = df[df["classe"] == "S_intent"]
    ambos = (si["frederico"] == "conceitual") & (si["davi"] == "conceitual")
    frac = float(ambos.mean()) if len(si) else None
    return {"n_fichas": len(df), "n_S_intent": len(si),
            "fracao_S_intent_conceitual_pelos_dois": frac,
            "atende": None if frac is None else frac >= limiar, "limiar": limiar,
            "kappa_cohen": float(cohen_kappa_score(df["frederico"], df["davi"])),
            "concordancia_bruta": float((df["frederico"] == df["davi"]).mean()),
            "S_lex_lexical_pelos_dois": float(((df["classe"] == "S_lex") & (df["frederico"] == "lexical")
                                              & (df["davi"] == "lexical")).sum())}


def _md(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def explicacao(layer: int, sae: str, f: int, fetch: Callable[[int, str, int], dict], tentativas: int = 3) -> str:
    import time
    for k in range(tentativas):
        try:
            return resumo_neuronpedia(fetch(layer, sae, f))["explicacao"]
        except Exception:  # rede instável: tenta de novo
            time.sleep(2 * (k + 1))
    return "(falha ao buscar no Neuronpedia)"


def provisorio_md(prov: dict, csv_dir: Path, sae: str, fetch: Callable[[int, str, int], dict] = fetch_feature) -> str:
    """Texto do `provisorio.md`: resumo das 8 combinações e o top-20 de cada lado com o rótulo do Neuronpedia."""
    linhas = [
        "# Modo provisório nos pares originais (C1 provisório)",
        "",
        "> **Diagnóstico, fora do veredito** (pré-registro, regra 4 da seção 5). Nos pares originais, só o estilo do "
        "texto já separa nocivo de benigno com AUROC 0,88 (`results/frente-d/baseline_lexical.json`), então features "
        "\"de intenção\" aqui podem ser features de estilo ou de assunto. Serve para depurar o código e ter uma "
        "primeira ideia. Só a descoberta (train + val).",
        f"> Gerado por `python -m featgrpo.interp.autointerp provisorio` a partir de `provisorio_{sae}.json` "
        f"(commit {str(prov.get('git_commit') or '')[:7]}). Os rótulos são as explicações automáticas do Neuronpedia "
        "(`gemma-3-4b-it/{L}-gemmascope-2-res-" + sae + "`), escritas por um LLM a partir de texto genérico: são "
        "ponto de partida, não verdade.",
        "",
        "Intenção provisória = ativação em Q⁺ − ativação em Q⁰, pareada por grupo. \"Efeito grande\" = "
        "\\|d\\| ≥ 0,5 e q < 0,01 (BH na família das 8 combinações). \"Sobrevivem ao estilo\" = controle de estilo "
        "(`DESVIOS.md`, 2026-11-21).",
        "",
        "## Resumo",
        "",
        "| Combinação | Pares na descoberta | Efeito grande | Mais ativas em Q⁺ | Mais ativas em Q⁰ | Sobrevivem ao estilo | R² do estilo (mediana) |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in prov["combinacoes"]:
        e = c.get("controle_estilo") or {}
        n_pos, n_neg = len(c["top20_mais_ativas_em_Q_harm"]), len(c["top20_mais_ativas_em_Q_border"])
        sob = f"{e['n_sobrevivem']} de {e['n_features']}" if e.get("status") == "ok" else "—"
        r2 = f"{e['r2_mediano']:.2f}".replace(".", ",") if e.get("status") == "ok" else "—"
        linhas.append(f"| L{c['camada']} · {c['rotulo']} | {c['n_grupos_descoberta']} | {c['n_sig_d05']} | "
                      f"{n_pos}{'+' if n_pos == 20 else ''} | {n_neg}{'+' if n_neg == 20 else ''} | {sob} | {r2} |")
    linhas += ["", "\"20+\" = há mais de 20 features desse lado; abaixo aparecem só as 20 de maior \\|d\\|.", ""]
    for c in prov["combinacoes"]:
        layer, pos = c["camada"], c["posicao"]
        df = pd.read_csv(Path(csv_dir) / f"provisorio_L{layer}_{pos}_{sae}.csv").set_index("feature")
        linhas += [f"## L{layer} · {c['rotulo']}", ""]
        for titulo, feats in (("Mais ativas em Q⁺ (nocivo)", c["top20_mais_ativas_em_Q_harm"]),
                              ("Mais ativas em Q⁰ (benigno)", c["top20_mais_ativas_em_Q_border"])):
            linhas += [f"**{titulo}**", ""]
            if not feats:
                linhas += ["Nenhuma feature com efeito grande.", ""]
                continue
            linhas += ["| # | Feature | d | Ativa em | Rótulo do Neuronpedia |", "|---|---|---|---|---|"]
            for k, f in enumerate(feats, 1):
                url = NP_PAGE.format(model=NP_MODEL, source=np_source(layer, sae), index=f)
                d = f"{df.loc[f, 'd_int']:.2f}".replace(".", ",").replace("-", "−")
                fr = f"{100 * df.loc[f, 'freq_ativacao']:.0f}%"
                linhas.append(f"| {k} | [{f}]({url}) | {d} | {fr} | {_md(explicacao(layer, sae, f, fetch))} |")
            linhas.append("")
    return "\n".join(linhas).rstrip() + "\n"


RECUSA = __import__("re").compile(r"refus|cannot|declin|disclaim|unable|can't|won't", __import__("re").I)
LABELS_POS = {"inst_last": "fim_pedido", "post_last": "pre_resposta"}


def _n(x: float, casas: int = 2) -> str:
    return f"{x:.{casas}f}".replace(".", ",").replace("-", "−")


def _lista(dir_: Path, cls: str, camada: int, pos: str, sae: str) -> set[int] | None:
    p = Path(dir_) / "listas" / f"{cls}_L{camada}_{pos}_{sae}.json"
    return set(json.loads(p.read_text(encoding="utf-8"))["features"]) if p.exists() else None


def _max_dl_slot(out_dir: Path) -> float | None:
    """Maior |d_L| no `slot_forma` (posições secundárias), para a checagem de sanidade da seção 1, regra 4."""
    vals = [pd.read_csv(c)["d_L"].abs().max() for c in sorted((Path(out_dir) / "secundarias").glob("feature_classes_L*_slot_forma_16k.csv"))]
    return float(max(vals)) if vals else None


def descoberta_md(c1: dict, out_dir: Path, fetch: Callable[[int, str, int], dict] = fetch_feature,
                  diag: dict | None = None, top: int = 10) -> str:
    """Texto do `descoberta.md`: o C1 no 2×2, só na descoberta. Primeiro o resumo em palavras, depois poucas
    tabelas pequenas; os detalhes ficam no anexo e nos CSV/JSON."""
    r = c1["16k"]
    p = r["principal"]
    L, pos = int(p["camada"]), p["posicao"]
    tag = f"L{L} · {p['rotulo']}"
    out_dir = Path(out_dir)
    combos = r["combinacoes_descoberta"]
    n_si = next(c["S_intent"] for c in combos if (c["camada"], c["posicao"]) == (L, pos))
    n_262 = (c1.get("262k") or {}).get("C1", {}).get("S_intent")
    n_lex = sum(c["S_lex"] for c in combos)
    n_mix = sum(c["S_mixed"] for c in combos)
    e, t = r.get("C1_controle_estilo") or {}, r.get("C3_controle_trig_only") or {}
    dom = r.get("por_dominio") or {}
    base = _lista(out_dir, "S_intent", L, pos, "16k") or set()
    rob = {nome: _lista(out_dir / f"robustez_{nome}", "S_intent", L, pos, "16k")
           for nome in ("qc_grammatical_sem_validador", "todos", "qc_pass_lenient")}
    df = pd.read_csv(out_dir / f"feature_classes_L{L}_{pos}_16k.csv") if (out_dir / f"feature_classes_L{L}_{pos}_16k.csv").exists() else None

    def vazamento(layer: int, posicao: str) -> tuple[float, float, int] | None:
        csv = out_dir / f"feature_classes_L{layer}_{posicao}_16k.csv"
        if not csv.exists():
            return None
        d = pd.read_csv(csv)
        big = d[(d["d_I"].abs() >= 0.5) & (d["q_I"] < 0.01)]
        if not len(big):
            return None
        return (float((np.sign(big["d_L"]) == np.sign(big["d_I"])).mean()), float((big["d_L"] / big["d_I"]).median()),
                int((big["classe"] != "S_intent").sum()))

    vz = vazamento(L, pos)
    fora = None
    if df is not None:
        fora = df[(df["d_I"].abs() >= 0.5) & (df["q_I"] < 0.01) & (df["classe"] != "S_intent")]
        fora = fora.reindex(fora["d_I"].abs().sort_values(ascending=False).index)

    rep, rep262 = r.get("replicacao_S_intent") or {}, (c1.get("262k") or {}).get("replicacao_S_intent") or {}
    aberta = bool(r.get("confirmacao_aberta")) and bool(rep)
    linhas = [
        "# C1 no 2×2: resultados" if aberta else "# C1 no 2×2: o que a descoberta mostra",
        "",
        ("> Descoberta (train + val) e confirmação (`test`, aberta uma vez, na combinação principal). "
         if aberta else "> Só a descoberta (train + val). A confirmação (`test`) ainda não foi aberta, e o C1 só fecha depois dela. ")
        + 
        f"Gerado por `python -m featgrpo.interp.autointerp descoberta` (commit {str(r.get('git_commit') or '')[:7]}, "
        f"conjunto `{r['subconjunto']}`). Os rótulos são explicações automáticas do Neuronpedia: ponto de partida, "
        "não verdade.",
        "",
        "## Em resumo",
        "",
        f"1. **Existem features que separam a intenção da palavra.** Na combinação principal ({tag}) são **{n_si}** "
        f"S_intent: reagem ao fato de o pedido ser nocivo e não à troca do gatilho. No SAE maior (262k), {n_262}.",
        f"2. **Elas são robustas.** Quase as mesmas aparecem com outros filtros de qualidade; "
        + (f"{e['n_sobrevivem']} de {e['n_features']} sobrevivem ao controle de estilo; " if e.get("status") == "ok" else "")
        + (f"{_n(100 * t['S_intent']['fracao_sem_efeito'], 0)}% não reagem ao gatilho solto (C3 atende); "
           if t.get("status") == "ok" else "")
        + (f"{dom['S_intent_gerais']} de {len(base)} valem nos 4 domínios" if dom else "")
        + (f"; e {min(v['S_intent_com_mesmo_sinal_e_d_ge_0_3'] for v in r['por_tipo_gatilho']['tipos'].values())} "
           f"de {r['por_tipo_gatilho']['n_S_intent']} se mantêm em cada um dos {len(r['por_tipo_gatilho']['tipos'])} tipos de gatilho."
           if r.get("por_tipo_gatilho") else "."),
        f"3. **Não existe feature \"da palavra\".** {n_lex} S_lex e {n_mix} S_mixed nas 8 combinações. A troca do "
        "gatilho muda a representação, mas cada gatilho mexe em features diferentes, e nenhuma muda do mesmo jeito "
        "na maioria dos grupos." + (f" A checagem de sanidade do pré-registro (seção 1, regra 4), que esperava efeito "
        f"lexical forte no próprio token do gatilho, **não passa como foi escrita**: lá o maior \\|d_L\\| é {_n(san)}."
        if (san := _max_dl_slot(out_dir)) is not None else ""),
    ]
    if vz:
        linhas.append(f"4. **Mas a palavra vaza um pouco nas features de intenção** (exploratório). Em {_n(100 * vz[0], 0)}% "
                      f"delas, o gatilho empurra no mesmo sentido do pedido nocivo, com cerca de {_n(100 * vz[1], 0)}% do "
                      "tamanho do efeito de intenção.")
    n_rec = 0
    if fora is not None and len(fora):
        n_rec = sum(bool(RECUSA.search(explicacao(L, "16k", int(f), fetch))) for f in fora["feature"].head(8))
        linhas.append(f"5. **As features de intenção mais fortes ficam fora de S_intent** (exploratório). São {len(fora)} "
                      "com efeito de intenção grande e esse vazamento pequeno do gatilho, que passa da margem de ±0,2 da "
                      f"regra. Entre as 8 mais fortes, {n_rec} têm rótulo de recusa no Neuronpedia.")
    if aberta:
        c1v, c4v = r.get("C1") or {}, (c1.get("262k") or {}).get("C1") or {}
        linhas.append(
            f"6. **Elas se confirmam em dados novos.** No `test` com gatilho inédito ({rep.get('n_gatilho_inedito')} grupos), "
            f"{rep.get('n_replicam')} de {rep.get('n_features')} replicam (mesmo sinal, \\|d_I\\| ≥ 0,3): "
            f"**C1 {c1v.get('status', '—')}**. No 262k, {rep262.get('n_replicam', '—')} de {rep262.get('n_features', '—')}: "
            f"**C4a {c4v.get('status', '—')}**.")
    cega = c1.get("C3_avaliacao_cega") or {}
    if cega:
        pr = cega["avaliador_principal"]
        n_c = round(cega["fracao_S_intent_conceitual"][pr] * cega["n_S_intent"])
        quem = "um LLM em contexto novo" if pr == "llm" else pr
        linhas.append(f"7. **Lidas às cegas, quase todas são conceituais.** {n_c} de {cega['n_S_intent']} fichas do top-20 "
                      f"foram julgadas conceituais por {quem} (mínimo 60%): **C3 {'atende' if cega['atende'] else 'não atende'}**"
                      + (" (conferência humana pendente)." if pr == "llm" and len(cega["fracao_S_intent_conceitual"]) == 1 else "."))
    tot_pos = {q: sum(c["S_intent"] for c in combos if c["posicao"] == q) for q in ("inst_last", "post_last")}
    onde = max(tot_pos, key=tot_pos.get)
    leitura = (f"**Leitura:** o modelo representa a intenção nociva separada da palavra, sobretudo no "
               f"{LABELS_POS[onde]}" if n_si >= 5 else "**Leitura:** há poucas features de intenção separadas da palavra")
    if vz and vz[0] >= 0.8 and vz[1] > 0:
        leitura += (", mas a palavra ainda dá um empurrão pequeno na mesma direção. Se esse empurrão explica o "
                    "over-refusal é a pergunta do C4b/c (frente D) e do C5 (frente B).")
    else:
        leitura += "."
    linhas += ["", leitura, ""]

    # onde está o sinal
    linhas += ["## Onde está o sinal", "", "Número de S_intent (SAE 16k):", "",
               "| Camada | fim_pedido | pre_resposta |", "|---|---|---|"]
    por = {(c["camada"], c["posicao"]): c["S_intent"] for c in combos}
    for layer in sorted({c["camada"] for c in combos}):
        a, b = por.get((layer, "inst_last"), "—"), por.get((layer, "post_last"), "—")
        if (layer, "inst_last") == (L, pos):
            a = f"**{a}** (principal)"
        if (layer, "post_last") == (L, pos):
            b = f"**{b}** (principal)"
        linhas.append(f"| L{layer} | {a} | {b} |")
    tot = {q: sum(v for (_, pp), v in por.items() if pp == q) for q in ("inst_last", "post_last")}
    mais, menos = ("post_last", "inst_last") if tot["post_last"] >= tot["inst_last"] else ("inst_last", "post_last")
    serie = [por[(layer, mais)] for layer in sorted({c["camada"] for c in combos}) if (layer, mais) in por]
    cresce = all(b >= a for a, b in zip(serie, serie[1:]))
    linhas += ["", f"Somando as camadas: {tot[mais]} S_intent no **{LABELS_POS[mais]}** contra {tot[menos]} no "
               f"{LABELS_POS[menos]}" + (", e o número cresce com a profundidade." if cresce else "."), ""]

    # as features
    if r.get("top20"):
        feats = r["top20"]["S_intent"]
        pos_f = [f for f in feats if f["d_I"] > 0][:top]
        neg_f = [f for f in feats if f["d_I"] < 0][:top]
        linhas += [f"## O que as S_intent parecem ser ({tag})", ""]
        for titulo, fs_ in (("Mais ativas nos pedidos **nocivos**", pos_f), ("Mais ativas nos pedidos **benignos**", neg_f)):
            if not fs_:
                continue
            linhas += [titulo, "", "| Feature | d_I | Rótulo do Neuronpedia |", "|---|---|---|"]
            for f in fs_:
                url = NP_PAGE.format(model=NP_MODEL, source=np_source(L, "16k"), index=f["feature"])
                linhas.append(f"| [{f['feature']}]({url}) | {_n(f['d_I'])} | {_md(explicacao(L, '16k', f['feature'], fetch))} |")
            linhas.append("")
        linhas += ["Sem rótulo = o Neuronpedia não tem explicação para a feature.", ""]
    if fora is not None and len(fora):
        linhas += [f"## As features de intenção mais fortes ficaram fora de S_intent ({tag})", "",
                   "| Feature | d_I | d_L | Rótulo do Neuronpedia |", "|---|---|---|---|"]
        for f in fora.head(8).itertuples():
            url = NP_PAGE.format(model=NP_MODEL, source=np_source(L, "16k"), index=int(f.feature))
            linhas.append(f"| [{int(f.feature)}]({url}) | {_n(f.d_I)} | {_n(f.d_L)} | "
                          f"{_md(explicacao(L, '16k', int(f.feature), fetch))} |")
        d_si = max((abs(f["d_I"]) for f in r["top20"]["S_intent"]), default=float("nan")) if r.get("top20") else float("nan")
        linhas += ["", f"{n_rec} das 8 têm rótulo de recusa. O maior d_I aqui é {_n(float(fora['d_I'].abs().max()))}, "
                   f"contra {_n(d_si)} da S_intent mais forte. O d_L delas é o vazamento do gatilho (item 4 do resumo). "
                   "No CSV aparecem como \"irrelevante\", porque a regra do pré-registro não as classifica.", ""]

    if aberta and df is not None and "d_I_test_inedito" in df.columns:
        si = df[df["classe"] == "S_intent"]
        tt = si["d_I_test_inedito"]
        linhas += ["## Confirmação (`test` com gatilho inédito)", "",
                   "| Medida | Valor |", "|---|---|",
                   f"| S_intent que replicam | {rep.get('n_replicam')} de {rep.get('n_features')} |",
                   f"| d_I no test ÷ d_I na descoberta (mediana) | {_n(float((tt / si['d_I']).median()))} |",
                   f"| Menor \\|d_I\\| no test | {_n(float(tt.abs().min()))} |",
                   f"| \\|d_L\\| no test (mediana) | {_n(float(si['d_L_test_inedito'].abs().median()))} |", "",
                   "O efeito de intenção no `test` tem o mesmo tamanho que na descoberta, e o efeito do gatilho continua "
                   "pequeno." if float((tt / si["d_I"]).median()) >= 0.8 else
                   "O efeito de intenção no `test` é menor que na descoberta.", ""]

    # anexo
    linhas += ["---", "", "## Anexo", ""]
    linhas += ["**A. Robustez (mesma combinação).** S_intent em cada conjunto; entre parênteses, quantas também são S_intent "
               "no principal.", ""]
    nomes = {"qc_grammatical_sem_validador": "sem o validador", "todos": "conjunto completo",
             "qc_pass_lenient": "`qc_pass_lenient` (1/3 dos grupos)"}
    linhas += [f"- principal (`qc_grammatical` + validador): {len(base)}"]
    for nome, s in rob.items():
        if s is not None:
            linhas.append(f"- {nomes[nome]}: {len(s)} ({len(s & base)})")
    for rc, rot in (("tamanho_pareado", "C4(c), células de tamanho pareado"), ("q0_retrieved", "C4(b), Q⁰ *retrieved*")):
        cp = out_dir / f"c4_{rc}" / "c1_c3.json"
        s = _lista(out_dir / f"c4_{rc}", "S_intent", L, pos, "16k")
        if cp.exists() and s is not None:
            rr = json.loads(cp.read_text(encoding="utf-8"))["16k"]
            rep_c = rr.get("replicacao_S_intent") or {}
            st = (rr.get("C1") or {}).get("status", "—")
            motivo = (rr.get("C1") or {}).get("motivo")
            linhas.append(f"- {rot}: {len(s)} ({len(s & base)}); replicam {rep_c.get('n_replicam', '—')} de "
                          f"{rep_c.get('n_features', '—')}; C1 {st}" + (f" ({motivo})" if motivo else ""))
    linhas.append("")
    if t.get("status") == "ok":
        ts = r.get("C3_controle_trig_only_sensibilidade_reamostragem_frase") or {}
        linhas += [f"**B. Controle `trig_only` (C3).** {t['n_pares_controle']} pares de frases em {t['n_grupos_2x2']} grupos; "
                   f"IC por grupo: {_n(100 * t['S_intent']['fracao_sem_efeito'], 0)}% das S_intent sem efeito; IC por frase "
                   f"(sensibilidade): {_n(100 * ts.get('S_intent', {}).get('fracao_sem_efeito', float('nan')), 0)}%. Sem S_lex, "
                   "o controle positivo (S_lex reagem ao gatilho solto) não pode ser verificado.", ""]
    if diag:
        linhas += ["**C. Por que não há S_lex** (`diagnostico_lexical_16k.json`). *Consistência*: 1 = o efeito aponta "
                   f"na mesma direção em todos os grupos; perto de 0 = cada grupo para um lado. Camada {L}:", "",
                   "| Posição | Intenção | Gatilho |", "|---|---|---|"]
        for c in diag["combinacoes"]:
            if c["camada"] == L:
                linhas.append(f"| {c['rotulo']} | {_n(c['consistencia_I'])} | {_n(c['consistencia_L'])} |")
        linhas += ["", "Mesmo no token do gatilho (`slot_forma`), onde a troca muda muito a representação, o efeito do "
                   "gatilho não tem direção comum entre os grupos. As outras camadas estão no JSON. Pela regra 4 da seção 1, "
                   "a falta de efeito lexical forte ali significaria que \"o SAE não representa o gatilho\"; o diagnóstico "
                   "mostra outra coisa: as features mudam com o gatilho, mas cada gatilho mexe em features diferentes. A "
                   "segunda parte da regra (efeito de intenção zero no `slot_forma`) não se aplica ao plano B, que não tem "
                   "lacuna de intenção: as linhas nociva e benigna são frases diferentes antes do gatilho.", ""]
    linhas += ["**D. Vazamento do gatilho por camada (pre_resposta, exploratório).** Entre as features com intenção grande: "
               "% com d_L no mesmo sentido de d_I, e o tamanho típico d_L ÷ d_I.", "",
               "| Camada | Mesmo sentido | d_L ÷ d_I |", "|---|---|---|"]
    for layer in sorted({c["camada"] for c in combos}):
        v = vazamento(layer, "post_last")
        if v:
            linhas.append(f"| L{layer} | {_n(100 * v[0], 0)}% | {_n(v[1])} |")
    altos = [layer for layer in sorted({c["camada"] for c in combos})
             if (v := vazamento(layer, "post_last")) and v[0] >= 0.8]
    linhas += ["", f"Camadas com ≥ 80% no mesmo sentido: {', '.join(f'L{x}' for x in altos) or 'nenhuma'}.", ""]
    linhas += ["**E. Arquivos.** Todas as features, com ICs e q: `feature_classes_L{L}_{pos}_{sae}.csv`. Listas para as "
               "frentes B e D: `listas/`. Figura: `fig_dI_dL_16k.png`.", ""]
    return "\n".join(linhas).rstrip() + "\n"


def lista_exploratoria(c1: dict, out_dir: Path, fetch: Callable[[int, str, int], dict] = fetch_feature,
                       sae: str = "16k") -> dict:
    """Exploratório, fora do pré-registro: features com efeito de intenção grande (|d_I| >= 0,5, q < 0,01) que
    ficaram fora de S_intent na combinação principal porque o IC de d_L passa de ±0,2. Só a descoberta. Cada uma
    leva o rótulo do Neuronpedia e a marca `rotulo_de_recusa` (busca simples no rótulo)."""
    p = c1["16k"]["principal"]
    L, pos = int(p["camada"]), p["posicao"]
    df = pd.read_csv(Path(out_dir) / f"feature_classes_L{L}_{pos}_{sae}.csv")
    fora = df[(df["d_I"].abs() >= 0.5) & (df["q_I"] < 0.01) & (df["classe"] != "S_intent")]
    fora = fora.reindex(fora["d_I"].abs().sort_values(ascending=False).index)
    detalhe = []
    for f in fora.itertuples():
        rot = explicacao(L, sae, int(f.feature), fetch)
        detalhe.append({"feature": int(f.feature), "d_I": round(float(f.d_I), 4), "d_L": round(float(f.d_L), 4),
                        "d_L_ci": [round(float(f.d_L_ci_lo), 4), round(float(f.d_L_ci_hi), 4)],
                        "mais_ativa_em": "nocivos" if f.d_I > 0 else "benignos", "rotulo_neuronpedia": rot,
                        "rotulo_de_recusa": bool(RECUSA.search(rot))})
    return {"features": [d["feature"] for d in detalhe], "classe": "intencao_forte_fora_de_S_intent",
            "exploratorio": True, "camada": L, "posicao": pos, "rotulo": LABELS_POS.get(pos, pos), "sae": sae,
            "criterio": "|d_I| >= 0,5 e q_I < 0,01 na descoberta, fora de S_intent (IC de d_L passa de ±0,2)",
            "ordenadas_por": "|d_I| decrescente", "n_rotulo_de_recusa": sum(d["rotulo_de_recusa"] for d in detalhe),
            "detalhe": detalhe}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fichas cegas do C3 (frente C).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pv = sub.add_parser("provisorio", help="monta results/frente-c/provisorio.md com os rótulos do Neuronpedia")
    pv.add_argument("--sae", default="16k")
    pv.add_argument("--dir", type=Path, default=OUT_DIR)
    ds = sub.add_parser("descoberta", help="monta results/frente-c/descoberta.md (C1 do 2×2, descoberta)")
    ds.add_argument("--dir", type=Path, default=OUT_DIR)
    ex = sub.add_parser("exploratoria", help="lista exploratória: intenção forte fora de S_intent (para a frente B)")
    ex.add_argument("--dir", type=Path, default=OUT_DIR)
    m = sub.add_parser("montar")
    m.add_argument("--root", type=Path, default=cache_io.DEFAULT_ROOT)
    m.add_argument("--sae", default="16k")
    m.add_argument("--cells", type=Path, default=PHASE1 / "cells_2x2.jsonl")
    m.add_argument("--out-dir", type=Path, default=OUT_DIR)
    m.add_argument("--chave-dir", type=Path, default=KEY_DIR)
    a = sub.add_parser("avaliar")
    a.add_argument("--llm", type=Path, default=None, help="rótulos do avaliador LLM (o principal; DESVIOS 2026-11-21)")
    a.add_argument("--frederico", type=Path, default=None)
    a.add_argument("--davi", type=Path, default=None)
    a.add_argument("--chave", type=Path, default=KEY_DIR / "fichas_chave.csv")
    a.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    if args.cmd == "provisorio":
        prov = json.loads((args.dir / f"provisorio_{args.sae}.json").read_text(encoding="utf-8"))
        out = args.dir / "provisorio.md"
        out.write_text(provisorio_md(prov, args.dir, args.sae), encoding="utf-8")
        print(f"salvo em {out}")
        return 0

    if args.cmd == "descoberta":
        c1 = json.loads((args.dir / "c1_c3.json").read_text(encoding="utf-8"))
        dp = args.dir / "diagnostico_lexical_16k.json"
        diag = json.loads(dp.read_text(encoding="utf-8")) if dp.exists() else None
        out = args.dir / "descoberta.md"
        out.write_text(descoberta_md(c1, args.dir, diag=diag), encoding="utf-8")
        print(f"salvo em {out}")
        return 0

    if args.cmd == "exploratoria":
        c1 = json.loads((args.dir / "c1_c3.json").read_text(encoding="utf-8"))
        res = lista_exploratoria(c1, args.dir)
        out = args.dir / "listas" / f"exploratorio_intencao_forte_L{res['camada']}_{res['posicao']}_{res['sae']}.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(res['features'])} features ({res['n_rotulo_de_recusa']} com rótulo de recusa) em {out}")
        return 0

    if args.cmd == "montar":
        resumo = json.loads((args.out_dir / "c1_c3.json").read_text(encoding="utf-8"))[args.sae]
        p = resumo["principal"]
        z = cache_io.load_latents(args.root, "cells2x2", int(p["camada"]), p["posicao"], args.sae)
        rows = cache_io.load_index(args.root, "cells2x2")
        fichas, chave = montar_fichas(resumo, args.sae, z, rows, textos_2x2(args.cells))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        args.chave_dir.mkdir(parents=True, exist_ok=True)
        fichas.to_csv(args.out_dir / "fichas_cegas.csv", index=False)
        chave.to_csv(args.chave_dir / "fichas_chave.csv", index=False)
        print(f"{len(fichas)} fichas em {args.out_dir / 'fichas_cegas.csv'}; chave em {args.chave_dir / 'fichas_chave.csv'}")
        return 0

    avaliadores = {nome: pd.read_csv(caminho) for nome, caminho in
                   (("llm", args.llm), ("frederico", args.frederico), ("davi", args.davi)) if caminho is not None}
    if not avaliadores:
        raise SystemExit("passe pelo menos um avaliador (--llm, --frederico, --davi)")
    res = avaliar_varios(avaliadores, pd.read_csv(args.chave), principal=next(iter(avaliadores)))
    path = args.out_dir / "c1_c3.json"
    resumo = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    resumo["C3_avaliacao_cega"] = res
    path.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
