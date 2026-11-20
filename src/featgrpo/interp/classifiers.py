"""Dissociação dupla com features do SAE (frente D, D4/D5): o critério C2 e os recortes C4b/c.

Entradas:
- latentes do 2×2 da frente B (`cells2x2`, uma camada × posição × SAE);
- as listas S_intent e S_lex da frente C, em `results/frente-c/listas/S_{intent,lex}_L{L}_{pos}_{sae}.json`
  (uma lista de índices de feature, ou `{"features": [...]}`);
- os textos de `datasets/phase1/cells_2x2.jsonl`, para o TF-IDF no mesmo protocolo.

O que calcula (sempre no `test`, com o recorte `gatilho_inedito` como principal):

- **Matriz C2:** para S_intent, S_lex, features aleatórias do mesmo tamanho (20 sorteios) e o
  probe denso (teto): AUROC de intenção com gatilho, intenção sem gatilho e presença do gatilho.
  Dois classificadores: L1 (`l1`) e média das features padronizadas (`media`). O `l1` decide.
- **C2(a), mínimo:** S_intent ≥ 0,85 com gatilho e ≥ 0,80 sem.
- **C2(b), principal:** treina em `*_trig`, testa em `*_notrig`. Razão = queda(S_intent) / queda(TF-IDF).
  **Atende** se razão ≤ 0,5 **e** o limite superior do IC 95% < 1 (S_intent cai menos que o TF-IDF
  com confiança). **Indefinido** se o TF-IDF cair ≤ 0,02 (o 2×2 não tirou o sinal lexical).
- **C2(c), consistência:** S_intent no gatilho ≤ 0,65; S_lex no gatilho ≥ 0,85. Não decide nada.
- **Recortes C4:** (b) Q⁰ original *retrieved*; (c) células com diferença ≤ 20% em tokens.

Uso:
    PYTHONPATH=src python -m featgrpo.interp.classifiers --layer 22 --pos post --sae 16k \
        --out results/frente-d/c2_c4.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from featgrpo.interp import cache_io
from featgrpo.interp.baselines import git_commit, make_tfidf, report
from featgrpo.interp.data import PHASE1, REPO
from featgrpo.interp.probes import fit_probe, has_trig, is_harm

SEED = 0
C_GRID = (0.01, 0.1, 1.0, 10.0)
N_RANDOM = 20
MIN_FREQ = 0.05
TFIDF_MIN_DROP = 0.02
TRIG, NOTRIG = ("harm_trig", "ben_trig"), ("harm_notrig", "ben_notrig")


# ---------------------------------------------------------------- classificadores restritos

class MeanScore:
    """Média das features padronizadas, com o sinal de cada uma tirado do treino."""

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        self.sign = np.sign(Z[y == 1].mean(0) - Z[y == 0].mean(0))
        self.sign[self.sign == 0] = 1
        return self

    def score(self, X):
        return (((X - self.mu) / self.sd) * self.sign).mean(1)


class L1:
    """Regressão logística L1 com C por GroupKFold (grupos = gatilho normalizado)."""

    def fit(self, X, y, groups):
        n = min(5, len(set(groups)))
        best, best_auc = C_GRID[-1], -1.0
        for C in C_GRID:
            aucs = []
            for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
                if len(set(y[te])) < 2:
                    continue
                m = self._make(C).fit(X[tr], y[tr])
                aucs.append(roc_auc_score(y[te], m.decision_function(X[te])))
            if aucs and np.mean(aucs) > best_auc:
                best, best_auc = C, float(np.mean(aucs))
        self.C = best
        self.m = self._make(best).fit(X, y)
        return self

    @staticmethod
    def _make(C):
        return LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C, max_iter=3000, random_state=SEED)

    def score(self, X):
        return self.m.decision_function(X)


def _fit_score(kind, Xtr, ytr, gtr, Xte):
    if kind == "l1":
        return L1().fit(Xtr, ytr, gtr).score(Xte)
    if kind == "media":
        return MeanScore().fit(Xtr, ytr).score(Xte)
    if kind == "denso":
        return fit_probe(Xtr, ytr, gtr)[0].predict_proba(Xte)[:, 1]
    raise ValueError(kind)


# ---------------------------------------------------------------- tarefas

def _mask(rows, split, cells=None, groups=None):
    return np.array([r["split"] == split and (cells is None or r["cell"] in cells)
                     and (groups is None or r["group_id"] in groups) for r in rows])


def _auc_report(scores, rows, label_fn, train_trigs, n_boot):
    y = np.array([label_fn(r) for r in rows])
    g = np.array([r["group_id"] for r in rows])
    seen = cache_io.seen_mask(rows, train_trigs)
    return {name: (report(scores[m], y[m], g[m], n_boot) if m.any() else None)
            for name, m in (("todos", np.ones(len(rows), bool)), ("gatilho_visto", seen), ("gatilho_inedito", ~seen))}


def task_matrix(X, rows, kind, n_boot, groups=None) -> dict:
    """AUROC de intenção (com/sem gatilho) e de gatilho, treinando no train e testando no test."""
    train_trigs = {r["trig_norm"] for r in rows if r["split"] == "train"}
    tr = _mask(rows, "train", groups=groups)
    gtr = np.array([r["trig_norm"] for r in rows])[tr]
    out = {}
    for task, label_fn, evals in (
        ("intencao", is_harm, {"com_gatilho": TRIG, "sem_gatilho": NOTRIG}),
        ("gatilho", has_trig, {"todas": None}),
    ):
        ytr = np.array([label_fn(r) for r, k in zip(rows, tr) if k])
        for name, cells in evals.items():
            te = _mask(rows, "test", cells, groups)
            s = _fit_score(kind, X[tr], ytr, gtr, X[te])
            out[f"{task}_{name}"] = _auc_report(s, [r for r, k in zip(rows, te) if k], label_fn, train_trigs, n_boot)
    return out


def _primary(m: dict, key: str):
    v = (m.get(key) or {}).get("gatilho_inedito") or {}
    return v.get("auroc")


# ---------------------------------------------------------------- C2(b): relativo ao TF-IDF

def relative_drop(score_trig, score_notrig, tf_trig, tf_notrig, y_trig, y_notrig, g_trig, g_notrig,
                  n_boot: int = 1000, seed: int = SEED) -> dict:
    """Razão queda(features)/queda(TF-IDF), com IC 95% por bootstrap sobre grupos (os mesmos nas duas células)."""
    def drops(idx_t, idx_n):
        a_t, a_n = roc_auc_score(y_trig[idx_t], score_trig[idx_t]), roc_auc_score(y_notrig[idx_n], score_notrig[idx_n])
        b_t, b_n = roc_auc_score(y_trig[idx_t], tf_trig[idx_t]), roc_auc_score(y_notrig[idx_n], tf_notrig[idx_n])
        return a_t - a_n, b_t - b_n

    all_t, all_n = np.arange(len(y_trig)), np.arange(len(y_notrig))
    d_feat, d_tf = drops(all_t, all_n)
    out = {"queda_features": round(d_feat, 4), "queda_tfidf": round(d_tf, 4)}
    if d_tf <= TFIDF_MIN_DROP:
        out.update(razao=None, ic95=None, atende=None,
                   status="indefinido: o TF-IDF não caiu sem o gatilho (o 2×2 não removeu o sinal lexical)")
        return out
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(g_trig) & set(g_notrig)))
    by_t = {g: np.flatnonzero(g_trig == g) for g in uniq}
    by_n = {g: np.flatnonzero(g_notrig == g) for g in uniq}
    ratios = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        it, in_ = np.concatenate([by_t[g] for g in pick]), np.concatenate([by_n[g] for g in pick])
        if len(set(y_trig[it])) < 2 or len(set(y_notrig[in_])) < 2:
            continue
        a, b = drops(it, in_)
        if b > 1e-6:
            ratios.append(a / b)
    ratio = d_feat / d_tf
    lo, hi = (float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))) if ratios else (None, None)
    ok = bool(ratio <= 0.5 and hi is not None and hi < 1.0)
    out.update(razao=round(ratio, 4), ic95=[round(lo, 4), round(hi, 4)] if lo is not None else None, atende=ok,
               status="atende" if ok else "não atende")
    return out


def c2b(X, rows, texts: dict | None, n_boot, groups=None) -> dict:
    """Treina intenção em *_trig (train), testa em *_trig e *_notrig (test, gatilho inédito)."""
    train_trigs = {r["trig_norm"] for r in rows if r["split"] == "train"}
    tr = _mask(rows, "train", TRIG, groups)
    ytr = np.array([is_harm(r) for r, k in zip(rows, tr) if k])
    gtr = np.array([r["trig_norm"] for r in rows])[tr]
    sel = {}
    for name, cells in (("trig", TRIG), ("notrig", NOTRIG)):
        te = _mask(rows, "test", cells, groups) & ~cache_io.seen_mask(rows, train_trigs)
        sel[name] = te
    clf = L1().fit(X[tr], ytr, gtr)
    s = {k: clf.score(X[m]) for k, m in sel.items()}
    y = {k: np.array([is_harm(r) for r, kk in zip(rows, m) if kk]) for k, m in sel.items()}
    g = {k: np.array([r["group_id"] for r, kk in zip(rows, m) if kk]) for k, m in sel.items()}
    if texts is None:
        return {"status": "pendente: textos do 2×2 (datasets/phase1/cells_2x2.jsonl) para o TF-IDF"}
    key = lambda r: (r["group_id"], r["cell"])  # noqa: E731
    tf = make_tfidf(C=10.0).fit([texts[key(r)] for r, k in zip(rows, tr) if k], ytr)
    t = {k: tf.predict_proba([texts[key(r)] for r, kk in zip(rows, m) if kk])[:, 1] for k, m in sel.items()}
    return relative_drop(s["trig"], s["notrig"], t["trig"], t["notrig"], y["trig"], y["notrig"],
                         g["trig"], g["notrig"], n_boot=n_boot)


# ---------------------------------------------------------------- C2 completo

def c2(Z, Xdense, rows, s_intent, s_lex, texts=None, n_boot=500, groups=None, n_random=N_RANDOM) -> dict:
    """Matriz C2 + controles + C2(a/b/c). `Z` = latentes densificados [N, n_features]."""
    tr = _mask(rows, "train", groups=groups)
    freq = (Z[tr] > 0).mean(0)
    pool = np.flatnonzero(freq >= MIN_FREQ)
    rng = np.random.default_rng(SEED)
    out = {"tamanhos": {"S_intent": len(s_intent), "S_lex": len(s_lex), "pool_aleatorio": int(len(pool))}, "matriz": {}}
    for name, feats in (("S_intent", s_intent), ("S_lex", s_lex)):
        if len(feats) == 0:
            out["matriz"][name] = None
            continue
        out["matriz"][name] = {k: task_matrix(Z[:, feats], rows, k, n_boot, groups) for k in ("l1", "media")}
    if len(s_intent) and len(pool) >= len(s_intent):
        draws = []
        for _ in range(n_random):
            feats = rng.choice(pool, size=len(s_intent), replace=False)
            m = task_matrix(Z[:, feats], rows, "l1", n_boot=0, groups=groups)
            draws.append({k: _primary(m, k) for k in m})
        out["matriz"]["aleatorio"] = {k: round(float(np.nanmean([d[k] for d in draws if d[k] is not None])), 4)
                                      for k in draws[0]}
    if Xdense is not None:
        out["matriz"]["denso"] = task_matrix(Xdense, rows, "denso", n_boot, groups)

    si = (out["matriz"].get("S_intent") or {}).get("l1") or {}
    sl = (out["matriz"].get("S_lex") or {}).get("l1") or {}
    a_com, a_sem = _primary(si, "intencao_com_gatilho"), _primary(si, "intencao_sem_gatilho")
    out["C2a"] = {"auroc_com_gatilho": a_com, "auroc_sem_gatilho": a_sem,
                  "atende": bool(a_com is not None and a_sem is not None and a_com >= 0.85 and a_sem >= 0.80)}
    out["C2b"] = c2b(Z[:, s_intent], rows, texts, n_boot, groups) if len(s_intent) else None
    g_int, g_lex = _primary(si, "gatilho_todas"), _primary(sl, "gatilho_todas")
    out["C2c_consistencia"] = {"S_intent_gatilho": g_int, "S_lex_gatilho": g_lex,
                               "ok": bool(g_int is not None and g_lex is not None and g_int <= 0.65 and g_lex >= 0.85),
                               "nota": "garantido em boa parte pela seleção das features; não decide o C2"}
    return out


# ---------------------------------------------------------------- recortes C4

def subsets(rows: list[dict], cells_meta: dict | None) -> dict:
    """Grupos dos recortes (b) Q⁰ retrieved e (c) células com diferença ≤ 20% em tokens."""
    out = {}
    if cells_meta:
        out["b_q0_retrieved"] = {g for g, m in cells_meta.items() if m.get("q_border_method") == "retrieved"}
    ntok: dict = {}
    for r in rows:
        if r.get("n_tokens") is not None:
            ntok.setdefault(r["group_id"], []).append(r["n_tokens"])
    out["c_tamanho_pareado"] = {g for g, v in ntok.items() if len(v) == 4 and (max(v) - min(v)) / max(v) <= 0.2}
    return out


# ---------------------------------------------------------------- execução

def load_list(path: Path) -> list[int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [int(i) for i in (data["features"] if isinstance(data, dict) else data)]


def load_cells_texts() -> tuple[dict, dict] | tuple[None, None]:
    p = PHASE1 / "cells_2x2.jsonl"
    if not p.exists():
        return None, None
    texts, meta = {}, {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                g = json.loads(line)
                meta[g["group_id"]] = g.get("orig") or {}
                for c, v in g["cells"].items():
                    texts[(g["group_id"], c)] = v["text"]
    return texts, meta


def run(root: Path, layer: int, pos: str, sae: str, lists_dir: Path, n_boot: int = 500) -> dict:
    rows = cache_io.load_index(root, "cells2x2")
    keep = np.array([r["cell"] is not None for r in rows])
    rows = [r for r, k in zip(rows, keep) if k]
    Z = cache_io.load_latents(root, "cells2x2", layer, pos, sae)[keep].toarray().astype(np.float32)
    dpath = cache_io.dense_path(root, "cells2x2", layer, pos)
    Xd = cache_io.load_dense(root, "cells2x2", layer, pos)[keep] if dpath.exists() else None
    tag = f"L{layer}_{pos}_{sae}"
    s_int = load_list(lists_dir / f"S_intent_{tag}.json")
    s_lex = load_list(lists_dir / f"S_lex_{tag}.json")
    texts, meta = load_cells_texts()

    res = {"principal": c2(Z, Xd, rows, s_int, s_lex, texts, n_boot)}
    for name, groups in subsets(rows, meta).items():
        res[name] = {"n_grupos": len(groups)}
        if len({r["group_id"] for r in rows if r["split"] == "test" and r["group_id"] in groups}) >= 20:
            res[name].update(c2(Z, None, rows, s_int, s_lex, texts, n_boot, groups=groups, n_random=0))
    return {
        "descricao": "C2 e recortes C4b/c (frente D). Ver docstring de src/featgrpo/interp/classifiers.py.",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "args": {"root": str(root), "layer": layer, "pos": pos, "sae": sae, "n_boot": n_boot, "seed": SEED,
                 "listas": str(lists_dir), "n_random": N_RANDOM, "min_freq": MIN_FREQ},
        "resultados": res,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="C2 (dissociação dupla) da frente D.")
    ap.add_argument("--root", type=Path, default=cache_io.DEFAULT_ROOT)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--sae", default="16k")
    ap.add_argument("--lists", type=Path, default=REPO / "results" / "frente-c" / "listas")
    ap.add_argument("--out", type=Path, default=Path("results/frente-d/c2_c4.json"))
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args(argv)
    res = run(args.root, args.layer, args.pos, args.sae, args.lists, args.n_boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    p = res["resultados"]["principal"]
    print(f"salvo em {args.out}\nC2a: {p['C2a']}\nC2b: {p['C2b']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
