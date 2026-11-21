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
- **C2(b), principal, relativo ao TF-IDF onde ele sofre** (emenda 3 da frente A): três generalizações,
  (1) gatilho inédito, (2) leave-one-domain-out e (3) XSTest. Em cada uma, razão = queda(S_intent) /
  queda(TF-IDF); **atende** se razão ≤ 0,5 e o limite superior do IC 95% < 1; **indefinido** se o
  TF-IDF cair ≤ 0,02. **O C2(b) é decidido só pelo XSTest** (regra votada em 2026-11-21, ver
  `results/DESVIOS.md`): no 2×2 real o TF-IDF não cai nas outras duas, que saem como descrição.
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

#
# Régua nova (emenda 3 do parecer da frente A, adotada em 2026-11-20): no molde, as palavras da
# intenção não mudam entre "com" e "sem gatilho", então o TF-IDF quase não cai ali. A comparação é
# feita onde o TF-IDF sofre, em três generalizações. Em cada uma:
#     queda = AUROC(dentro da distribuição) − AUROC(fora)       (tarefa: intenção, as 4 células)
#     razão = queda(S_intent) / queda(TF-IDF)
#     atende  se razão ≤ 0,5 e limite superior do IC 95% < 1;  indefinido se queda(TF-IDF) ≤ 0,02
# O C2(b) é decidido **só pelo XSTest** (votado em 2026-11-21, DESVIOS). Gatilho inédito e
# leave-one-domain-out continuam calculados, como descrição: no 2×2 real o TF-IDF cai ≤ 0,02 nelas.

GENERALIZACOES = ("gatilho_inedito", "leave_one_domain_out", "xstest")
DECIDE = "xstest"


def relative_drop_folds(folds: list[dict], n_boot: int = 1000, seed: int = SEED) -> dict:
    """Razão das quedas (features/TF-IDF), média sobre as dobras, com IC 95% por bootstrap sobre grupos.

    Cada dobra: `feat_in`, `feat_out`, `tf_in`, `tf_out` (notas), `y_in`, `y_out`, `g_in`, `g_out`.
    No bootstrap, os grupos de dentro e de fora de cada dobra são sorteados com reposição.
    """
    def drops(idx):
        df, dt = [], []
        for f, (ii, io) in zip(folds, idx):
            df.append(roc_auc_score(f["y_in"][ii], f["feat_in"][ii]) - roc_auc_score(f["y_out"][io], f["feat_out"][io]))
            dt.append(roc_auc_score(f["y_in"][ii], f["tf_in"][ii]) - roc_auc_score(f["y_out"][io], f["tf_out"][io]))
        return float(np.mean(df)), float(np.mean(dt))

    full = [(np.arange(len(f["y_in"])), np.arange(len(f["y_out"]))) for f in folds]
    d_feat, d_tf = drops(full)
    out = {"n_dobras": len(folds), "queda_features": round(d_feat, 4), "queda_tfidf": round(d_tf, 4)}
    if d_tf <= TFIDF_MIN_DROP:
        out.update(razao=None, ic95=None, atende=None,
                   status="indefinido: o TF-IDF não caiu nesta generalização")
        return out

    rng = np.random.default_rng(seed)

    def by_group(g):
        uniq = np.array(sorted(set(g)))
        return uniq, {k: np.flatnonzero(g == k) for k in uniq}

    index = [(by_group(f["g_in"]), by_group(f["g_out"])) for f in folds]
    ratios = []
    for _ in range(n_boot):
        idx, ok = [], True
        for f, ((ui, mi), (uo, mo)) in zip(folds, index):
            ii = np.concatenate([mi[g] for g in rng.choice(ui, size=len(ui))])
            io = np.concatenate([mo[g] for g in rng.choice(uo, size=len(uo))])
            if len(set(f["y_in"][ii])) < 2 or len(set(f["y_out"][io])) < 2:
                ok = False
                break
            idx.append((ii, io))
        if not ok:
            continue
        a, b = drops(idx)
        if b > 1e-6:
            ratios.append(a / b)
    ratio = d_feat / d_tf
    lo, hi = (float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))) if ratios else (None, None)
    ok = bool(ratio <= 0.5 and hi is not None and hi < 1.0)
    out.update(razao=round(ratio, 4), ic95=[round(lo, 4), round(hi, 4)] if lo is not None else None, atende=ok,
               status="atende" if ok else "não atende")
    return out


def _fold(Z, texts_of, rows, tr, ins, outs, out_Z=None, out_rows=None, out_texts=None) -> dict | None:
    """Treina S_intent (L1) e TF-IDF em `tr`; nota dentro (`ins`) e fora (`outs`, ou um conjunto externo)."""
    ytr = np.array([is_harm(rows[i]) for i in tr])
    if len(tr) == 0 or len(set(ytr)) < 2:
        return None
    gtr = np.array([rows[i]["trig_norm"] for i in tr])
    clf = L1().fit(Z[tr], ytr, gtr)
    tf = make_tfidf(C=10.0).fit([texts_of(rows[i]) for i in tr], ytr)
    y_in = np.array([is_harm(rows[i]) for i in ins])
    if out_Z is None:
        out_Z, out_rows, out_texts = Z[outs], [rows[i] for i in outs], [texts_of(rows[i]) for i in outs]
        y_out = np.array([is_harm(r) for r in out_rows])
    else:
        y_out = np.array([int(r.get("label") in ("harmful", 1)) for r in out_rows])
    if len(set(y_in)) < 2 or len(set(y_out)) < 2:
        return None
    return {"feat_in": clf.score(Z[ins]), "feat_out": clf.score(out_Z),
            "tf_in": tf.predict_proba([texts_of(rows[i]) for i in ins])[:, 1], "tf_out": tf.predict_proba(out_texts)[:, 1],
            "y_in": y_in, "y_out": y_out,
            "g_in": np.array([rows[i]["group_id"] for i in ins]), "g_out": np.array([r["group_id"] for r in out_rows])}


def c2b(Z, rows, texts: dict | None, n_boot, groups=None, xstest: tuple | None = None) -> dict:
    """C2(b) nas três generalizações. `Z` já restrito às features de S_intent.

    `xstest` = (Z_xstest, linhas do índice do xstest com `label`, textos na mesma ordem), quando existir.
    """
    if texts is None:
        return {"status": "pendente: textos do 2×2 (datasets/phase1/cells_2x2.jsonl) para o TF-IDF", "atende": None}
    texts_of = lambda r: texts[(r["group_id"], r["cell"])]  # noqa: E731
    ok_g = (lambda r: True) if groups is None else (lambda r: r["group_id"] in groups)
    idx = lambda f: np.flatnonzero([f(r) and ok_g(r) for r in rows])  # noqa: E731
    train_trigs = {r["trig_norm"] for r in rows if r["split"] == "train"}
    res = {}

    # 1. gatilho inédito: dentro = test com gatilho visto; fora = test com gatilho inédito
    f = _fold(Z, texts_of, rows, idx(lambda r: r["split"] == "train"),
              idx(lambda r: r["split"] == "test" and r["trig_norm"] in train_trigs),
              idx(lambda r: r["split"] == "test" and r["trig_norm"] not in train_trigs))
    res["gatilho_inedito"] = relative_drop_folds([f], n_boot) if f else {"status": "sem dados", "atende": None}

    # 2. leave-one-domain-out: treina em 3 domínios; dentro = test dos 3; fora = test do 4º
    folds = []
    for d in sorted({r["domain"] for r in rows}):
        f = _fold(Z, texts_of, rows, idx(lambda r: r["split"] == "train" and r["domain"] != d),
                  idx(lambda r: r["split"] == "test" and r["domain"] != d),
                  idx(lambda r: r["split"] == "test" and r["domain"] == d))
        if f:
            folds.append(f)
    res["leave_one_domain_out"] = relative_drop_folds(folds, n_boot) if len(folds) >= 2 else \
        {"status": "sem dados: precisa de pelo menos 2 domínios", "atende": None}

    # 3. XSTest: treina no 2×2; dentro = test do 2×2; fora = XSTest
    if xstest is None:
        res["xstest"] = {"status": "pendente: cache e textos do XSTest", "atende": None}
    else:
        Zx, xs_rows, xs_texts = xstest
        f = _fold(Z, texts_of, rows, idx(lambda r: r["split"] == "train"), idx(lambda r: r["split"] == "test"),
                  None, out_Z=Zx, out_rows=xs_rows, out_texts=xs_texts)
        res["xstest"] = relative_drop_folds([f], n_boot) if f else {"status": "sem dados", "atende": None}

    n_ok = sum(1 for k in GENERALIZACOES if res[k].get("atende") is True)
    n_def = sum(1 for k in GENERALIZACOES if res[k].get("atende") is not None)
    res["n_atende"] = n_ok
    res["n_definidas"] = n_def
    res["atende"] = res[DECIDE].get("atende")
    res["descritivas"] = [k for k in GENERALIZACOES if k != DECIDE]
    res["regra"] = ("decidido só pelo XSTest: razão queda(S_intent)/queda(TF-IDF) ≤ 0,5 e IC sup. < 1; "
                    "gatilho inédito e leave-one-domain-out são descrição (votado em 2026-11-21)")
    return res


# ---------------------------------------------------------------- C2 completo

def c2(Z, Xdense, rows, s_intent, s_lex, texts=None, n_boot=500, groups=None, n_random=N_RANDOM,
       xstest: tuple | None = None) -> dict:
    """Matriz C2 + controles + C2(a/b/c). `Z` = latentes densificados [N, n_features].

    `xstest` = (Z_xstest com todas as features, linhas com `label` e `group_id`, textos), opcional.
    """
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
    xs = (xstest[0][:, s_intent], xstest[1], xstest[2]) if xstest is not None and len(s_intent) else None
    out["C2b"] = c2b(Z[:, s_intent], rows, texts, n_boot, groups, xs) if len(s_intent) else None
    g_int, g_lex = _primary(si, "gatilho_todas"), _primary(sl, "gatilho_todas")
    out["C2c_consistencia"] = {"S_intent_gatilho": g_int, "S_lex_gatilho": g_lex,
                               "ok": (None if not len(s_lex) else
                                      bool(g_int is not None and g_lex is not None and g_int <= 0.65 and g_lex >= 0.85)),
                               "nota": ("indefinido: S_lex vazia (a frente C não achou feature lexical consistente)"
                                        if not len(s_lex) else
                                        "garantido em boa parte pela seleção das features; não decide o C2")}
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


def load_xstest(root: Path, layer: int, pos: str, sae: str) -> tuple | None:
    """(latentes, linhas com label, textos) do XSTest, juntando o cache da frente B com o arquivo da frente A."""
    from featgrpo.interp.baselines import load_xstest_items

    items = load_xstest_items()
    if items is None or not (Path(root) / "index" / "xstest.jsonl").exists() \
            or not cache_io.latents_path(root, "xstest", layer, pos, sae).exists():
        return None
    by_id = {it["group_id"]: it for it in items}
    rows = cache_io.load_index(root, "xstest")
    keep = np.array([r["group_id"] in by_id for r in rows])
    rows = [{**r, "label": by_id[r["group_id"]]["label"]} for r, k in zip(rows, keep) if k]
    Zx = cache_io.load_latents(root, "xstest", layer, pos, sae)[keep].toarray().astype(np.float32)
    return Zx, rows, [by_id[r["group_id"]]["text"] for r in rows]


def run(root: Path, layer: int, pos: str, sae: str, lists_dir: Path, n_boot: int = 500,
        subset: str = "principal") -> dict:
    from featgrpo.interp.probes import failed_groups, subset_mask

    rows = cache_io.load_index(root, "cells2x2")
    keep = np.array([r["cell"] is not None for r in rows]) & subset_mask(rows, subset, failed_groups())
    rows = [r for r, k in zip(rows, keep) if k]
    Zs = cache_io.load_latents(root, "cells2x2", layer, pos, sae)[keep]
    dpath = cache_io.dense_path(root, "cells2x2", layer, pos)
    Xd = cache_io.load_dense(root, "cells2x2", layer, pos)[keep] if dpath.exists() else None
    tag = f"L{layer}_{pos}_{sae}"
    s_int = load_list(lists_dir / f"S_intent_{tag}.json")
    s_lex = load_list(lists_dir / f"S_lex_{tag}.json")
    # Só as colunas usadas viram densas: o pool aleatório (freq ≥ MIN_FREQ no train) e as listas.
    # No 262k, densificar tudo daria ~7 GB. O resultado é o mesmo, porque o c2 só olha essas colunas.
    tr = np.array([r["split"] == "train" for r in rows])
    freq = np.asarray((Zs[tr] > 0).mean(0)).ravel()
    cols = np.array(sorted(set(np.flatnonzero(freq >= MIN_FREQ)) | set(s_int) | set(s_lex)), dtype=int)
    pos_of = {c: i for i, c in enumerate(cols)}
    Z = Zs[:, cols].toarray().astype(np.float32)
    s_int, s_lex = [pos_of[c] for c in s_int], [pos_of[c] for c in s_lex]
    texts, meta = load_cells_texts()
    xstest = load_xstest(root, layer, pos, sae)
    if xstest is not None:
        xstest = (xstest[0][:, cols], xstest[1], xstest[2])

    res = {"principal": c2(Z, Xd, rows, s_int, s_lex, texts, n_boot, xstest=xstest)}
    for name, groups in subsets(rows, meta).items():
        res[name] = {"n_grupos": len(groups)}
        if len({r["group_id"] for r in rows if r["split"] == "test" and r["group_id"] in groups}) >= 20:
            res[name].update(c2(Z, None, rows, s_int, s_lex, texts, n_boot, groups=groups, n_random=0))
    return {
        "descricao": "C2 e recortes C4b/c (frente D). Ver docstring de src/featgrpo/interp/classifiers.py.",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "args": {"root": str(root), "layer": layer, "pos": pos, "sae": sae, "n_boot": n_boot, "seed": SEED,
                 "subconjunto": subset, "n_grupos": len({r["group_id"] for r in rows}),
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
    ap.add_argument("--subset", default="principal", choices=("principal", "qc_grammatical", "todos"))
    args = ap.parse_args(argv)
    res = run(args.root, args.layer, args.pos, args.sae, args.lists, args.n_boot, args.subset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    p = res["resultados"]["principal"]
    print(f"salvo em {args.out}\nC2a: {p['C2a']}\nC2b: {p['C2b']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
