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


def top_itens_2x2(z: sp.csr_matrix, rows: list[dict], feats: list[int], textos: dict, k: int = 10) -> dict[int, list[str]]:
    """Para cada feature, os k itens do 2×2 (só descoberta) com maior ativação, com a célula."""
    keep = [i for i, r in enumerate(rows) if r.get("cell") and r.get("split") in DISCOVERY]
    sub = z[keep][:, feats].toarray()
    out = {}
    for j, f in enumerate(feats):
        order = [i for i in np.argsort(-sub[:, j])[:k] if sub[i, j] > 0]
        out[f] = [f"{rows[keep[i]]['cell']}: {_limpa(textos.get((rows[keep[i]]['group_id'], rows[keep[i]]['cell']), '(texto não encontrado)'))}"
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fichas cegas do C3 (frente C).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("montar")
    m.add_argument("--root", type=Path, default=cache_io.DEFAULT_ROOT)
    m.add_argument("--sae", default="16k")
    m.add_argument("--cells", type=Path, default=PHASE1 / "cells_2x2.jsonl")
    m.add_argument("--out-dir", type=Path, default=OUT_DIR)
    m.add_argument("--chave-dir", type=Path, default=KEY_DIR)
    a = sub.add_parser("avaliar")
    a.add_argument("--frederico", type=Path, required=True)
    a.add_argument("--davi", type=Path, required=True)
    a.add_argument("--chave", type=Path, default=KEY_DIR / "fichas_chave.csv")
    a.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

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

    res = avaliar(pd.read_csv(args.frederico), pd.read_csv(args.davi), pd.read_csv(args.chave))
    path = args.out_dir / "c1_c3.json"
    resumo = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    resumo["C3_avaliacao_cega"] = res
    path.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
