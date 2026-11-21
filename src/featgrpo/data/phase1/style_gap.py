"""Checagem da manipulação do caminho (b): o estilo sozinho ainda separa a linha nociva da benigna?

Classificador só de estilo (os 14 atributos de `interp.baselines.style_features`, regressão logística),
AUROC nociva × benigna em GroupKFold por grupo (as 4 células de um grupo ficam no mesmo fold).
Compara o 2×2 original (`cells_2x2.jsonl`) com o reescrito em estilo comum, nos mesmos grupos.

    python -m featgrpo.data.phase1.style_gap data/phase1/cells_style_smoke.jsonl [--out results/frente-a/estilo.json]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ...interp.baselines import style_features
from .common1 import out_dir

CELLS = ("harm_trig", "harm_notrig", "ben_trig", "ben_notrig")


def auroc(rows: list[dict], cells=CELLS) -> dict:
    texts, y, g = [], [], []
    for r in rows:
        for c in cells:
            texts.append(r["cells"][c]["text"]); y.append(int(c.startswith("harm"))); g.append(r["group_id"])
    X, y, g = style_features(texts), np.array(y), np.array(g)
    k = min(5, len(set(g)))
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(X, y, g):
        m = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=1.0, max_iter=2000))]).fit(X[tr], y[tr])
        scores[te] = m.predict_proba(X[te])[:, 1]
    lens = [abs(len(r["cells"]["harm_trig"]["text"].split()) - len(r["cells"]["ben_trig"]["text"].split())) for r in rows]
    return {"groups": len(rows), "auroc_style": round(float(roc_auc_score(y, scores)), 3),
            "len_diff_words_median": float(np.median(lens)),
            "same_last_char": round(float(np.mean([r["cells"]["harm_trig"]["text"].strip()[-1:] ==
                                                   r["cells"]["ben_trig"]["text"].strip()[-1:] for r in rows])), 3)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("styled")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    sty = [json.loads(l) for l in open(a.styled)]
    ids = {r["group_id"] for r in sty}
    orig = [json.loads(l) for l in open(out_dir() / "cells_2x2.jsonl")]
    orig_same = [r for r in orig if r["group_id"] in ids]
    rep = {"original_todos": auroc(orig), "original_mesmos_grupos": auroc(orig_same) if len(orig_same) >= 10 else None,
           "estilo_comum": auroc(sty)}
    for k in ("qc_grammatical",):
        sub = [r for r in sty if r.get(k)]
        if len(sub) >= 10:
            rep[f"estilo_comum_{k}"] = auroc(sub)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(rep, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    sys.exit(main())
