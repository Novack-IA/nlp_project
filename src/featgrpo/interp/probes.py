"""Âncora densa da frente D (D3): probes lineares no residual completo.

Responde: a informação de intenção está no residual e sobrevive à troca do gatilho?
Se nem o probe denso separar, o problema está nos dados ou no modelo, não no SAE.

Para cada camada × posição do cache:

- **2×2** (`cells2x2`, o que vale para o veredito):
  - `intencao`: treina em todas as células do `train`, testa no `test` (com e sem gatilho);
  - `cruzado_trig_para_notrig`: treina em `harm_trig` × `ben_trig`, testa em `harm_notrig` × `ben_notrig`;
  - `cruzado_notrig_para_trig`: a volta;
  - `gatilho_independe_intencao`: treina gatilho em `ben_trig` × `ben_notrig`, testa em `harm_trig` × `harm_notrig`;
  - `direcoes`: cosseno entre a direção de intenção, a de gatilho e a de recusa (pares, à la Arditi).
- **Pares originais** (`pairs`, só diagnóstico): `intencao_pares` treina no `train` e testa em `val`/`test`.
- **XSTest** (`xstest`, quando existir): probe dos pares aplicado fora da distribuição.

Todo `test` sai em `todos`, `gatilho_visto` e `gatilho_inedito`. O C é escolhido só no `train`,
com GroupKFold por gatilho normalizado.

Uso:
    PYTHONPATH=src python -m featgrpo.interp.probes --root artifacts/acts/gemma-3-4b-it \
        --out results/frente-d/probes.json --fig results/frente-d/probes_auroc_por_camada.png
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from featgrpo.interp import cache_io
from featgrpo.interp.baselines import git_commit, report

SEED = 0
C_GRID = (0.001, 0.01, 0.1, 1.0)


# ---------------------------------------------------------------- núcleo

def make_probe(C: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, random_state=SEED))


def fit_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    """Regressão logística L2 com C escolhido por GroupKFold (grupos = gatilho normalizado)."""
    n_splits = min(n_splits, len(set(groups)))
    best, best_auc = C_GRID[0], -1.0
    if n_splits >= 2:
        for C in C_GRID:
            aucs = []
            for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
                if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
                    continue
                m = make_probe(C).fit(X[tr], y[tr])
                aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
            if aucs and np.mean(aucs) > best_auc:
                best, best_auc = C, float(np.mean(aucs))
    return make_probe(best).fit(X, y), best


def split_reports(scores: np.ndarray, y: np.ndarray, rows: list[dict], train_trigs: set, n_boot: int) -> dict:
    """Relatório em todos / gatilho visto / gatilho inédito."""
    groups = np.array([r["group_id"] for r in rows])
    seen = cache_io.seen_mask(rows, train_trigs)
    out = {}
    for name, m in (("todos", np.ones(len(rows), bool)), ("gatilho_visto", seen), ("gatilho_inedito", ~seen)):
        out[name] = report(scores[m], y[m], groups[m], n_boot) if m.any() else None
    return out


def experiment(X, rows, train_sel, test_sel, label_fn, train_trigs, n_boot) -> dict:
    """Treina em `train_sel`, testa em `test_sel`; rótulo por `label_fn(row)`."""
    tr = np.flatnonzero([train_sel(r) for r in rows])
    te = np.flatnonzero([test_sel(r) for r in rows])
    y_tr = np.array([label_fn(rows[i]) for i in tr])
    y_te = np.array([label_fn(rows[i]) for i in te])
    if len(tr) == 0 or len(te) == 0 or len(set(y_tr)) < 2:
        return {"erro": "sem dados suficientes"}
    g_tr = np.array([rows[i]["trig_norm"] for i in tr])
    model, C = fit_probe(X[tr], y_tr, g_tr)
    scores = model.predict_proba(X[te])[:, 1]
    return {"C": C, "n_train": int(len(tr)), "test": split_reports(scores, y_te, [rows[i] for i in te], train_trigs, n_boot)}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def is_harm(r):
    return int(r["cell"].startswith("harm"))


def has_trig(r):
    return int(r["cell"].endswith("_trig"))


def cells_experiments(X: np.ndarray, rows: list[dict], n_boot: int, refusal_dir: np.ndarray | None = None) -> dict:
    """Os experimentos do 2×2 numa camada × posição."""
    keep = np.array([r["cell"] is not None for r in rows])
    X, rows = X[keep], [r for r, k in zip(rows, keep) if k]
    train_trigs = {r["trig_norm"] for r in rows if r["split"] == "train"}

    def sel(split, cells=None):
        return lambda r: r["split"] == split and (cells is None or r["cell"] in cells)

    trig_cells, notrig_cells = ("harm_trig", "ben_trig"), ("harm_notrig", "ben_notrig")
    out = {
        "intencao": experiment(X, rows, sel("train"), sel("test"), is_harm, train_trigs, n_boot),
        "intencao_com_gatilho": experiment(X, rows, sel("train"), sel("test", trig_cells), is_harm, train_trigs, n_boot),
        "intencao_sem_gatilho": experiment(X, rows, sel("train"), sel("test", notrig_cells), is_harm, train_trigs, n_boot),
        "cruzado_trig_para_notrig": experiment(X, rows, sel("train", trig_cells), sel("test", notrig_cells),
                                               is_harm, train_trigs, n_boot),
        "cruzado_notrig_para_trig": experiment(X, rows, sel("train", notrig_cells), sel("test", trig_cells),
                                               is_harm, train_trigs, n_boot),
        "gatilho": experiment(X, rows, sel("train"), sel("test"), has_trig, train_trigs, n_boot),
        "gatilho_independe_intencao": experiment(X, rows, sel("train", ("ben_trig", "ben_notrig")),
                                                 sel("test", ("harm_trig", "harm_notrig")), has_trig, train_trigs, n_boot),
    }
    tr = np.array([r["split"] == "train" for r in rows])
    h = np.array([is_harm(r) for r in rows], bool)
    t = np.array([has_trig(r) for r in rows], bool)
    d_int = X[tr & h].mean(0) - X[tr & ~h].mean(0)
    d_lex = X[tr & t].mean(0) - X[tr & ~t].mean(0)
    out["direcoes"] = {"cos_intencao_gatilho": round(cosine(d_int, d_lex), 4)}
    if refusal_dir is not None:
        out["direcoes"]["cos_intencao_recusa"] = round(cosine(d_int, refusal_dir), 4)
        out["direcoes"]["cos_gatilho_recusa"] = round(cosine(d_lex, refusal_dir), 4)
    return out


def refusal_direction(X: np.ndarray, rows: list[dict]) -> np.ndarray:
    """Média de Q⁺ menos média de Q⁰ nos pares do train (direção de recusa à la Arditi)."""
    tr = np.array([r["split"] == "train" for r in rows])
    h = np.array([r["item"] == "q_harm" for r in rows])
    return X[tr & h].mean(0) - X[tr & ~h].mean(0)


def pairs_experiment(X: np.ndarray, rows: list[dict], n_boot: int):
    """Probe de intenção nos pares originais (diagnóstico). Devolve o relatório e o modelo treinado."""
    train_trigs = {r["trig_norm"] for r in rows if r["split"] == "train"}
    y = np.array([int(r["item"] == "q_harm") for r in rows])
    tr = np.array([r["split"] == "train" for r in rows])
    model, C = fit_probe(X[tr], y[tr], np.array([r["trig_norm"] for r in rows])[tr])
    out = {"C": C}
    for split in ("val", "test"):
        m = np.array([r["split"] == split for r in rows])
        out[split] = split_reports(model.predict_proba(X[m])[:, 1], y[m], [r for r, k in zip(rows, m) if k],
                                   train_trigs, n_boot)
    return out, model


# ---------------------------------------------------------------- execução

def run(root: Path, n_boot: int = 500) -> dict:
    root = Path(root)
    res: dict = {"pairs": {}, "cells2x2": {}, "xstest": {}}
    pair_rows = cache_io.load_index(root, "pairs") if (root / "index" / "pairs.jsonl").exists() else None
    cell_rows = cache_io.load_index(root, "cells2x2") if (root / "index" / "cells2x2.jsonl").exists() else None
    xs_rows = cache_io.load_index(root, "xstest") if (root / "index" / "xstest.jsonl").exists() else None

    refusal = {}
    for L, pos in cache_io.available(root, "pairs") if pair_rows else []:
        X = cache_io.load_dense(root, "pairs", L, pos)
        rep, model = pairs_experiment(X, pair_rows, n_boot)
        res["pairs"][f"L{L}/{pos}"] = rep
        refusal[(L, pos)] = refusal_direction(X, pair_rows)
        if xs_rows and cache_io.dense_path(root, "xstest", L, pos).exists():
            Xx = cache_io.load_dense(root, "xstest", L, pos)
            yx = np.array([int(r.get("label") == "harmful" or r.get("label") == 1) for r in xs_rows])
            gx = np.array([r["group_id"] for r in xs_rows])
            res["xstest"][f"L{L}/{pos}"] = report(model.predict_proba(Xx)[:, 1], yx, gx, n_boot)

    for L, pos in cache_io.available(root, "cells2x2") if cell_rows else []:
        X = cache_io.load_dense(root, "cells2x2", L, pos)
        # direção de recusa vem dos pares na posição equivalente (slot_forma ↔ trig; slot_intencao não tem par)
        ref = refusal.get((L, {"slot_forma": "trig"}.get(pos, pos)))
        res["cells2x2"][f"L{L}/{pos}"] = cells_experiments(X, cell_rows, n_boot, ref)

    if not res["cells2x2"]:
        res["cells2x2"] = "pendente: cache cells2x2 ainda não existe (frente B, depois de A1)"
    return {
        "descricao": "Probes densos (frente D, D3). Ver docstring de src/featgrpo/interp/probes.py.",
        "leitura": "O veredito usa cells2x2 (gatilho inédito). 'pairs' é diagnóstico: o estilo sozinho já separa os pares.",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "args": {"root": str(root), "n_boot": n_boot, "seed": SEED, "C_grid": list(C_GRID)},
        "resultados": res,
    }


def plot(res: dict, path: Path, tfidf_auroc: float | None = None) -> None:
    """AUROC por camada (gatilho inédito): dentro da célula e generalização cruzada; TF-IDF tracejado."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = res["resultados"]["cells2x2"]
    if not isinstance(cells, dict) or not cells:
        return
    positions = sorted({k.split("/")[1] for k in cells})
    fig, axes = plt.subplots(1, len(positions), figsize=(4 * len(positions), 3.2), sharey=True, squeeze=False)
    series = [("intencao", "intenção (todas)"), ("cruzado_trig_para_notrig", "com → sem gatilho"),
              ("cruzado_notrig_para_trig", "sem → com gatilho"), ("gatilho_independe_intencao", "gatilho (ben → harm)")]
    for ax, pos in zip(axes[0], positions):
        layers = sorted(int(k.split("/")[0][1:]) for k in cells if k.endswith("/" + pos))
        for key, label in series:
            ys = []
            for L in layers:
                e = cells[f"L{L}/{pos}"][key]
                t = (e.get("test") or {}).get("gatilho_inedito") or {}
                ys.append(t.get("auroc"))
            ax.plot(layers, ys, marker="o", label=label)
        if tfidf_auroc is not None:
            ax.axhline(tfidf_auroc, ls="--", c="gray", label="TF-IDF (sem gatilho)")
        ax.set_title(pos)
        ax.set_xlabel("camada")
        ax.set_ylim(0.4, 1.0)
    axes[0][0].set_ylabel("AUROC (test, gatilho inédito)")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Probes densos da frente D.")
    ap.add_argument("--root", type=Path, default=cache_io.DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=Path("results/frente-d/probes.json"))
    ap.add_argument("--fig", type=Path, default=Path("results/frente-d/probes_auroc_por_camada.png"))
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args(argv)
    res = run(args.root, n_boot=args.n_boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(res, args.fig)
    print(f"salvo em {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
