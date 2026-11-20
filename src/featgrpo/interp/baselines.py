"""Baselines de texto da frente D (D1 / D-P1): a "régua" lexical do projeto.

Três classificadores de nocivo (1) × benigno (0), todos treinados só no split `train`:

- ``tfidf``: TF-IDF de palavras (1–2) + caracteres (3–5) e regressão logística.
  Mede quanto dá para acertar só olhando as palavras.
- ``style``: só atributos de estilo (tamanho, pontuação, forma da pergunta).
  Mede a confusão de fonte (PKU × gerador) nos pares originais.
- ``trigger_only``: só a presença do gatilho no texto. Nos pares originais
  os dois lados têm o gatilho, então deve dar AUROC ≈ 0,5; isso documenta o desenho.

Para cada conjunto (val, test e, quando existirem, células do 2×2 e XSTest),
reporta AUROC, acurácia e acurácia pareada, com IC de 95% por bootstrap
sobre grupos. O `val` e o `test` saem também divididos em gatilho **visto**
no treino e gatilho **inédito** (proposta, regra 1).

O hiperparâmetro C é escolhido só no `train`, com GroupKFold por gatilho
normalizado (gatilhos iguais nunca ficam em dobras diferentes).

Uso:
    PYTHONPATH=src python -m featgrpo.interp.baselines \
        --out results/frente-d/baseline_lexical.json
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from featgrpo.interp.data import PHASE1, REPO, load_pairs, prompt_items

SEED = 0
C_GRID = (0.1, 1.0, 10.0, 100.0)
CELLS = ("harm_trig", "harm_notrig", "ben_trig", "ben_notrig")

_STEM = PorterStemmer()
_TOKEN = re.compile(r"[a-z0-9']+")


# ---------------------------------------------------------------- gatilhos

def _stems(text: str) -> list[str]:
    return [_STEM.stem(t) for t in _TOKEN.findall(text.lower())]


def normalize_trigger(trigger: str) -> str:
    """Forma canônica do gatilho (minúsculas + radical Porter), para agrupar e comparar."""
    return " ".join(_stems(trigger))


def trigger_present(text: str, trigger: str) -> bool:
    """Gatilho no texto: literal (sem caixa) ou, se falhar, todos os radicais presentes."""
    if trigger.lower().strip() in text.lower():
        return True
    trig = set(_stems(trigger))
    return bool(trig) and trig <= set(_stems(text))


# ---------------------------------------------------------------- atributos de estilo

_WH = ("how", "what", "why", "where", "when", "which", "who", "can", "could", "is", "are", "do", "does", "should", "would")


def style_features(texts: list[str]) -> np.ndarray:
    """Atributos que não dependem do conteúdo: tamanho, pontuação e forma da pergunta."""
    rows = []
    for t in texts:
        words = t.split()
        n_words = max(len(words), 1)
        first = words[0].lower().strip("\"'(") if words else ""
        rows.append([
            len(t),
            len(words),
            sum(len(w) for w in words) / n_words,
            t.count(","),
            t.count("?"),
            t.count("."),
            t.count("!"),
            t.count(":") + t.count(";"),
            sum(c.isupper() for c in t) / max(len(t), 1),
            sum(c.isdigit() for c in t),
            len(re.findall(r"[.!?](\s|$)", t)),
            float(first in _WH),
            float(t.rstrip().endswith("?")),
            float(" i " in f" {t.lower()} " or " my " in f" {t.lower()} "),
        ])
    return np.asarray(rows, dtype=float)


# ---------------------------------------------------------------- modelos

def make_tfidf(C: float) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([("feats", feats), ("clf", LogisticRegression(C=C, max_iter=2000, random_state=SEED))])


def make_style(C: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=2000, random_state=SEED))])


def choose_C(make, X, y, groups, n_splits: int = 5) -> tuple[float, dict]:
    """C pela AUROC média em GroupKFold no train. Nada do val/test entra aqui."""
    cv = GroupKFold(n_splits=n_splits)
    scores = {}
    for C in C_GRID:
        fold = []
        for tr, te in cv.split(X, y, groups):
            m = make(C).fit(_take(X, tr), y[tr])
            fold.append(roc_auc_score(y[te], m.predict_proba(_take(X, te))[:, 1]))
        scores[C] = float(np.mean(fold))
    best = max(scores, key=scores.get)
    return best, {str(k): round(v, 4) for k, v in scores.items()}


def _take(X, idx):
    return [X[i] for i in idx] if isinstance(X, list) else X[idx]


# ---------------------------------------------------------------- métricas

def paired_accuracy(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Fração de grupos em que o item nocivo recebe nota maior que o benigno (empate = 0,5)."""
    by_g: dict = {}
    for s, lab, g in zip(scores, y, groups):
        by_g.setdefault(g, {}).setdefault(int(lab), []).append(s)
    wins = []
    for d in by_g.values():
        if 0 in d and 1 in d:
            h, b = np.mean(d[1]), np.mean(d[0])
            wins.append(1.0 if h > b else 0.5 if h == b else 0.0)
    return float(np.mean(wins)) if wins else float("nan")


def metrics(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    out = {"n_items": int(len(y)), "n_groups": int(len(set(groups)))}
    if len(set(y)) < 2:
        out.update(auroc=None, accuracy=None, paired_accuracy=None)
        return out
    out["auroc"] = float(roc_auc_score(y, scores))
    out["accuracy"] = float(accuracy_score(y, scores >= 0.5))
    out["paired_accuracy"] = paired_accuracy(scores, y, groups)
    return out


def bootstrap_ci(scores: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 n_boot: int = 1000, seed: int = SEED, alpha: float = 0.05) -> dict:
    """IC por bootstrap sobre grupos: sorteia grupos com reposição e leva todos os itens de cada um."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    stats = {"auroc": [], "accuracy": [], "paired_accuracy": []}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        yy = y[idx]
        if len(set(yy)) < 2:
            continue
        ss = scores[idx]
        # grupos repetidos recebem rótulos novos, para o pareado contar cada cópia
        gg = np.concatenate([np.full(len(idx_by_g[g]), k) for k, g in enumerate(pick)])
        stats["auroc"].append(roc_auc_score(yy, ss))
        stats["accuracy"].append(accuracy_score(yy, ss >= 0.5))
        stats["paired_accuracy"].append(paired_accuracy(ss, yy, gg))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if v else None
            for k, v in stats.items()}


def _round(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, list):
        return [_round(v) for v in x]
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return x


def report(scores, y, groups, n_boot) -> dict:
    m = metrics(scores, y, groups)
    if m["auroc"] is not None:
        m["ci95"] = bootstrap_ci(scores, y, groups, n_boot=n_boot)
    return _round(m)


# ---------------------------------------------------------------- conjuntos opcionais

def load_cells_items() -> list[dict] | None:
    """Itens das células do 2×2, se a frente A já entregou `cells_2x2.jsonl`."""
    path = PHASE1 / "cells_2x2.jsonl"
    if not path.exists():
        return None
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            for cell in CELLS:
                c = g["cells"][cell]
                out.append({"group_id": g["group_id"], "cell": cell, "split": g["split"],
                            "trigger": g["trigger"], "domain": g["domain"],
                            "label": int(cell.startswith("harm")), "text": c["text"]})
    return out


def load_xstest_items() -> list[dict] | None:
    path = PHASE1 / "external" / "xstest.jsonl"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [{"group_id": r["id"], "type": r.get("type"), "label": int(r["label"] == "harmful"),
             "text": r["text"]} for r in rows]


# ---------------------------------------------------------------- execução

def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def run(n_boot: int = 1000) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)

    items = prompt_items(load_pairs())
    for it in items:
        it["trig_norm"] = normalize_trigger(it["trigger"])
        it["trig_present"] = trigger_present(it["text"], it["trigger"])

    by_split = {s: [it for it in items if it["split"] == s] for s in ("train", "val", "test")}
    train = by_split["train"]
    train_trigs = {it["trig_norm"] for it in train}

    def arrays(rows):
        return ([r["text"] for r in rows], np.array([r["label"] for r in rows]),
                np.array([r["group_id"] for r in rows]))

    Xtr, ytr, _ = arrays(train)
    gtr_trig = np.array([r["trig_norm"] for r in train])

    # --- seleção de C só no train, GroupKFold por gatilho
    C_tfidf, cv_tfidf = choose_C(make_tfidf, Xtr, ytr, gtr_trig)
    Str = style_features(Xtr)
    C_style, cv_style = choose_C(make_style, Str, ytr, gtr_trig)

    tfidf = make_tfidf(C_tfidf).fit(Xtr, ytr)
    style = make_style(C_style).fit(Str, ytr)

    def score(model_name, rows):
        texts = [r["text"] for r in rows]
        if model_name == "tfidf":
            return tfidf.predict_proba(texts)[:, 1]
        if model_name == "style":
            return style.predict_proba(style_features(texts))[:, 1]
        if model_name == "trigger_only":
            # nota = presença do gatilho; 0,5 quando ausente/presente nos dois lados dá empate
            return np.array([1.0 if r.get("trig_present", trigger_present(r["text"], r["trigger"])) else 0.0
                             for r in rows])
        raise ValueError(model_name)

    def evaluate(rows, models=("tfidf", "style", "trigger_only")):
        _, y, g = arrays(rows)
        return {m: report(score(m, rows), y, g, n_boot) for m in models}

    results: dict = {"pairs": {}}
    for split in ("val", "test"):
        rows = by_split[split]
        seen = [r for r in rows if r["trig_norm"] in train_trigs]
        unseen = [r for r in rows if r["trig_norm"] not in train_trigs]
        results["pairs"][split] = {
            "all": evaluate(rows),
            "trigger_seen": evaluate(seen),
            "trigger_unseen": evaluate(unseen),
        }
        results["pairs"][split]["by_domain"] = {
            d: evaluate([r for r in rows if r["domain"] == d], models=("tfidf",))
            for d in sorted({r["domain"] for r in rows})
        }

    # presença do gatilho por lado: documenta que o "só gatilho" não separa os pares
    results["trigger_presence_rate"] = {
        item: round(float(np.mean([r["trig_present"] for r in items if r["item"] == item])), 4)
        for item in ("q_harm", "q_border")
    }

    # --- 2×2 e XSTest, quando existirem (o modelo segue o treinado nos pares)
    cells = load_cells_items()
    if cells is None:
        results["cells_2x2"] = "pendente: datasets/phase1/cells_2x2.jsonl ainda não existe (entrega da frente A)"
    else:
        test_cells = [c for c in cells if c["split"] == "test"]
        out = {}
        for name, (h, b) in {"com_gatilho": ("harm_trig", "ben_trig"),
                             "sem_gatilho": ("harm_notrig", "ben_notrig")}.items():
            rows = [c for c in test_cells if c["cell"] in (h, b)]
            out[name] = evaluate(rows, models=("tfidf", "style"))
        a, b_ = out["com_gatilho"]["tfidf"]["auroc"], out["sem_gatilho"]["tfidf"]["auroc"]
        out["queda_auroc_tfidf"] = round(a - b_, 4) if a is not None and b_ is not None else None
        results["cells_2x2"] = out

    xs = load_xstest_items()
    results["xstest"] = (evaluate(xs, models=("tfidf", "style")) if xs is not None
                         else "pendente: datasets/phase1/external/xstest.jsonl ainda não existe (entrega da frente A)")

    return {
        "descricao": "Baselines de texto (frente D, D-P1). Ver docstring de src/featgrpo/interp/baselines.py.",
        "leitura": ("Nos pares originais, estes números medem sobretudo a diferença de estilo entre as fontes "
                    "(PKU × gerador): são diagnóstico, não evidência para o veredito. A régua de C2/C4 é o "
                    "resultado nas células do 2×2."),
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "args": {"n_boot": n_boot, "seed": SEED, "C_grid": list(C_GRID), "cv": "GroupKFold(5) por gatilho normalizado, só train"},
        "versoes": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__},
        "dados": {
            "pares": len(items) // 2,
            "itens_por_split": dict(Counter(it["split"] for it in items)),
            "gatilhos_unicos_train": len(train_trigs),
            "pares_test_gatilho_visto": sum(1 for r in by_split["test"] if r["trig_norm"] in train_trigs) // 2,
            "pares_test_gatilho_inedito": sum(1 for r in by_split["test"] if r["trig_norm"] not in train_trigs) // 2,
        },
        "hiperparametros": {"tfidf": {"C": C_tfidf, "cv_auroc": cv_tfidf}, "style": {"C": C_style, "cv_auroc": cv_style}},
        "resultados": results,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "results" / "frente-d" / "baseline_lexical.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    res = run(n_boot=args.n_boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    t = res["resultados"]["pairs"]["test"]
    print(f"salvo em {args.out}")
    for part in ("all", "trigger_seen", "trigger_unseen"):
        print(part, {m: t[part][m]["auroc"] for m in ("tfidf", "style", "trigger_only")})


if __name__ == "__main__":
    sys.exit(main())
