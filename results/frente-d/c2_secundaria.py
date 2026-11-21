"""Frente D, análise secundária do C2 (davi-frente-d.md, itens 5 e 6).

1. C2 por domínio e por tipo de gatilho: S_intent (L1), features aleatórias do mesmo tamanho
   (mediana de 20 sorteios) e TF-IDF, todos treinados no `train` do conjunto principal e avaliados no
   `test` por domínio e por tipo de gatilho (intenção, as 4 células).
2. Figuras do paper: heatmap da matriz C2 e a queda no XSTest contra o controle aleatório.

Não decide nada; descreve onde o C2 vale. Uso (da raiz do repo):
    PYTHONPATH=src python results/frente-d/c2_secundaria.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from featgrpo.interp import cache_io
from featgrpo.interp import classifiers as C
from featgrpo.interp.baselines import make_tfidf, report
from featgrpo.interp.probes import failed_groups, subset_mask

ROOT = Path("artifacts/acts/gemma-3-4b-it")
OUT = Path("results/frente-d")
LISTS = Path("results/frente-c/listas")
L, POS, N_RANDOM, N_BOOT = 29, "post_last", 20, 500


def by_slice(rows, mask_test, scores, key):
    out = {}
    for v in sorted({r[key] for r, m in zip(rows, mask_test) if m}):
        sel = np.array([m and r[key] == v for r, m in zip(rows, mask_test)])
        idx = np.flatnonzero(sel)
        y = np.array([C.is_harm(rows[i]) for i in idx])
        g = np.array([rows[i]["group_id"] for i in idx])
        out[v] = report(scores[sel[mask_test]], y, g, N_BOOT) if len(set(y)) == 2 else {"n_items": int(len(idx))}
        out[v]["n_groups"] = int(len(set(g)))
    return out


def secundaria(sae: str) -> dict:
    rows = cache_io.load_index(ROOT, "cells2x2")
    keep = np.array([r["cell"] is not None for r in rows]) & subset_mask(rows, "principal", failed_groups())
    rows = [r for r, k in zip(rows, keep) if k]
    Zs = cache_io.load_latents(ROOT, "cells2x2", L, POS, sae)[keep]
    s_int = C.load_list(LISTS / f"S_intent_L{L}_{POS}_{sae}.json")
    tr = np.array([r["split"] == "train" for r in rows])
    te = np.array([r["split"] == "test" for r in rows])
    y = np.array([C.is_harm(r) for r in rows])
    g = np.array([r["trig_norm"] for r in rows])

    def l1_scores(cols):
        Z = Zs[:, cols].toarray().astype(np.float32)
        return C.L1().fit(Z[tr], y[tr], g[tr]).score(Z[te])

    texts, _ = C.load_cells_texts()
    tx = [texts[(r["group_id"], r["cell"])] for r in rows]
    tf = make_tfidf(C=10.0).fit([t for t, m in zip(tx, tr) if m], y[tr])
    s_tf = tf.predict_proba([t for t, m in zip(tx, te) if m])[:, 1]

    freq = np.asarray((Zs[tr] > 0).mean(0)).ravel()
    pool = np.setdiff1d(np.flatnonzero(freq >= C.MIN_FREQ), s_int)
    rng = np.random.default_rng(C.SEED)
    rand = [l1_scores(rng.choice(pool, size=len(s_int), replace=False)) for _ in range(N_RANDOM)]
    s_si = l1_scores(np.array(s_int))

    res = {}
    for key in ("domain", "trigger_type"):
        si, tfr = by_slice(rows, te, s_si, key), by_slice(rows, te, s_tf, key)
        rr = [by_slice(rows, te, s, key) for s in rand]
        res[key] = {v: {"S_intent": si[v], "tfidf": tfr[v],
                        "aleatorio_mediana_auroc": float(np.median([x[v].get("auroc") for x in rr]))}
                    for v in si}
    return res


def figuras(c2: dict, ctl: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.colors import LinearSegmentedColormap

    ink, muted, grid = "#1f1f1e", "#6b6a66", "#d9d8d3"
    blues = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "text.color": ink})

    # Figura 1: matriz C2 (AUROC no test, gatilho inédito)
    tasks = [("intencao_com_gatilho", "intenção\ncom gatilho"), ("intencao_sem_gatilho", "intenção\nsem gatilho"),
             ("gatilho_todas", "presença\ndo gatilho")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), sharey=True)
    for ax, sae in zip(axes, ("16k", "262k")):
        m = c2[sae]["resultados"]["principal"]["matriz"]
        names = [("S_intent", "S_intent (L1)"), ("S_intent_media", "S_intent (média)"),
                 ("aleatorio", "aleatório, mesmo tamanho"), ("denso", "probe denso")]
        M = []
        for key, _ in names:
            if key == "aleatorio":
                M.append([m["aleatorio"][t] for t, _ in tasks])
            else:
                src = m["S_intent"]["media" if key == "S_intent_media" else "l1"] if key.startswith("S_intent") else m["denso"]
                M.append([src[t]["gatilho_inedito"]["auroc"] for t, _ in tasks])
        M = np.array(M, float)
        ax.imshow(M, cmap=blues, vmin=0.5, vmax=1.0, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=9,
                        color="white" if M[i, j] > 0.8 else ink)
        ax.set_xticks(range(len(tasks)), [t for _, t in tasks])
        ax.set_yticks(range(len(names)), [n for _, n in names])
        ax.set_title(f"SAE {sae}", fontsize=10, color=ink)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Matriz do C2: AUROC no test com gatilho inédito (L29 · pré-resposta)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_c2_matriz.png", dpi=200)
    plt.close(fig)

    # Figura 2: queda de AUROC no XSTest (treinado no 2×2)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    labels, y0 = [], 0
    for sae in ("16k", "262k"):
        c = ctl[sae]
        rnd = [v for v in c["aleatorio_20"]["queda_features"] if v is not None]
        ax.scatter(rnd, [y0] * len(rnd), s=22, color="#86b6ef", zorder=2, label="aleatório (20 sorteios)" if y0 == 0 else None)
        ax.scatter([c["S_intent"]["queda_features"]], [y0], s=60, color="#0d366b", zorder=3,
                   label="S_intent" if y0 == 0 else None, edgecolor="white", linewidth=1.5)
        labels.append(f"SAE {sae}")
        y0 += 1
    ax.scatter([ctl["denso"]["queda_features"]], [y0], s=60, color="#6b6a66", zorder=3, label="probe denso")
    labels.append("residual (denso)")
    tf = ctl["16k"]["S_intent"]["queda_tfidf"]
    ax.axvline(tf, color="#c0392b", lw=2, zorder=1)
    ax.text(tf, 2.0, "TF-IDF cai " + f"{tf:.2f}".replace(".", ",") + " ", color="#c0392b", va="center", ha="right", fontsize=9)
    ax.axvline(tf / 2, color=muted, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(tf / 2, -0.75, " limite do C2(b) (metade)", color=muted, va="center", fontsize=8)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(-1.0, y0 + 0.5)
    ax.set_xlim(-0.01, 0.31)
    ax.set_xlabel("queda de AUROC: test do 2×2 → XSTest")
    ax.grid(axis="x", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_c2b_xstest.png", dpi=200)
    plt.close(fig)


def main() -> None:
    c2 = {s: json.loads((OUT / f"c2_c4_L29_post_last_{s}.json").read_text(encoding="utf-8")) for s in ("16k", "262k")}
    ctl = json.loads((OUT / "c2b_controle_aleatorio.json").read_text(encoding="utf-8"))
    figuras(c2, ctl)
    res = {"descricao": "C2 por domínio e por tipo de gatilho (análise secundária, não decide). test do conjunto principal, intenção nas 4 células.",
           "camada": L, "posicao": POS}
    for sae in ("16k", "262k"):
        res[sae] = secundaria(sae)
        print(sae, "ok", flush=True)
    (OUT / "c2_secundaria.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
