"""Figuras da frente C. Figura 2 do paper: d_I × d_L de todas as features ativas, por camada e posição.

Cada painel é uma combinação (camada × posição principal). Eixo x: efeito de intenção (d_I); eixo y: efeito
do gatilho (d_L). A faixa cinza é a zona de equivalência (±0,2) e as linhas tracejadas, o limiar de efeito
grande (±0,5). Se houver separação, as S_intent formam um "braço" no eixo x e as S_lex, outro no eixo y.

    PYTHONPATH=src python -m featgrpo.interp.figuras --sae 16k      # lê results/frente-c/feature_classes_*.csv

Cores: as 3 primeiras posições da paleta categórica de referência (skill de dataviz), validadas para
todos os pares em modo claro (ΔE CVD ≥ 9,2). O verde-água fica abaixo de 3:1 de contraste, então cada classe
também tem forma própria e a legenda é escrita.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from featgrpo.interp.data import REPO  # noqa: E402

LAYERS = (9, 17, 22, 29)
MAIN_POSITIONS = ("inst_last", "post_last")
LABELS = {"inst_last": "fim_pedido", "post_last": "pre_resposta"}
OUT_DIR = REPO / "results" / "frente-c"

SURFACE, INK, INK_2, GRID, NEUTRO = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc", "#b9b7b0"
CLASSES = {  # classe: (cor, marcador, rótulo)
    "S_intent": ("#2a78d6", "o", "S_intent (intenção, sem efeito do gatilho)"),
    "S_lex": ("#eb6834", "^", "S_lex (gatilho, sem efeito de intenção)"),
    "S_mixed": ("#1baf7a", "s", "S_mixed (os dois)"),
}


def _virgula(x, _pos=None) -> str:
    return f"{x:g}".replace(".", ",").replace("-", "−")


def scatter_di_dl(tabelas: dict, out_path: Path, sae: str, principal: tuple[int, str] | None = None,
                  margem_eq: float = 0.2, d_min: float = 0.5, lim_max: float = 4.0) -> Path:
    """Desenha a grade 2 posições × 4 camadas. `tabelas[(camada, posição)]` tem d_I, d_L e classe."""
    todos = pd.concat([t[["d_I", "d_L"]] for t in tabelas.values()], ignore_index=True).abs().stack()
    lim = float(min(lim_max, max(1.0, np.nanquantile(todos, 0.999) * 1.08))) if len(todos) else 1.0

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(len(MAIN_POSITIONS), len(LAYERS), figsize=(13, 6.8), sharex=True, sharey=True,
                             facecolor=SURFACE)
    for r, pos in enumerate(MAIN_POSITIONS):
        for c, layer in enumerate(LAYERS):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            ax.axvspan(-margem_eq, margem_eq, color=GRID, alpha=0.55, lw=0, zorder=0)
            ax.axhspan(-margem_eq, margem_eq, color=GRID, alpha=0.55, lw=0, zorder=0)
            for v in (-d_min, d_min):
                ax.axvline(v, color=INK_2, lw=0.6, ls=(0, (3, 3)), zorder=1)
                ax.axhline(v, color=INK_2, lw=0.6, ls=(0, (3, 3)), zorder=1)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.tick_params(length=0)
            ax.xaxis.set_major_formatter(FuncFormatter(_virgula))
            ax.yaxis.set_major_formatter(FuncFormatter(_virgula))
            for s in ax.spines.values():
                s.set_visible(False)
            df = tabelas.get((layer, pos))
            titulo = f"L{layer} · {LABELS[pos]}"
            if df is None or df.empty:
                ax.text(0, 0, "sem dados", ha="center", va="center", color=INK_2)
                ax.set_title(titulo, fontsize=9.5, color=INK, loc="left", pad=14)
                continue
            x, y = df["d_I"].clip(-lim, lim), df["d_L"].clip(-lim, lim)
            resto = ~df["classe"].isin(CLASSES)
            ax.scatter(x[resto], y[resto], s=3, color=NEUTRO, alpha=0.6, lw=0, rasterized=True, zorder=2)
            contagem = []
            for cls, (cor, marca, _) in CLASSES.items():
                m = df["classe"] == cls
                if m.any():
                    ax.scatter(x[m], y[m], s=22, color=cor, marker=marca, edgecolors=SURFACE, linewidths=0.6, zorder=3)
                contagem.append(f"{cls} {int(m.sum())}")
            ax.set_title(titulo, fontsize=9.5, color=INK, loc="left", pad=14,
                         fontweight="bold" if (layer, pos) == principal else None)
            ax.text(0, 1.012, "  ·  ".join(contagem), transform=ax.transAxes, va="bottom", fontsize=7.5, color=INK_2)
            if (layer, pos) == principal:
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_edgecolor(INK)
                    s.set_linewidth(1.2)
                ax.text(0.98, 0.03, "combinação principal", transform=ax.transAxes, ha="right", fontsize=7.5, color=INK)
    for ax in axes[-1, :]:
        ax.set_xlabel("d_I  (efeito de intenção)")
    for ax in axes[:, 0]:
        ax.set_ylabel("d_L  (efeito do gatilho)")
    handles = [Line2D([], [], marker=marca, ls="", color=cor, markeredgecolor=SURFACE, markersize=6, label=rot)
               for cor, marca, rot in CLASSES.values()]
    handles.append(Line2D([], [], marker="o", ls="", color=NEUTRO, markersize=4, label="sem classe"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Efeito de intenção × efeito do gatilho por feature (SAE {sae}, descoberta)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.925, f"Faixa cinza: |d| < {_virgula(margem_eq)} (zona de equivalência). Tracejado: "
                          f"|d| = {_virgula(d_min)}. Valores além de ±{_virgula(round(lim, 1))} aparecem na borda.",
             fontsize=8, color=INK_2)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.12, top=0.86, wspace=0.12, hspace=0.38)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Figura d_I × d_L da frente C.")
    ap.add_argument("--sae", default="16k")
    ap.add_argument("--dir", type=Path, default=OUT_DIR)
    ap.add_argument("--paper-dir", type=Path, default=OUT_DIR.parents[1] / "docs" / "paper" / "figuras",
                    help="onde copiar a figura 2 do paper (só no 16k)")
    args = ap.parse_args(argv)
    tabelas = {}
    for layer in LAYERS:
        for pos in MAIN_POSITIONS:
            p = args.dir / f"feature_classes_L{layer}_{pos}_{args.sae}.csv"
            if p.exists():
                tabelas[(layer, pos)] = pd.read_csv(p)
    principal = None
    resumo = args.dir / "c1_c3.json"
    if resumo.exists():
        pr = (json.loads(resumo.read_text(encoding="utf-8")).get(args.sae) or {}).get("principal")
        if pr:
            principal = (int(pr["camada"]), pr["posicao"])
    out = scatter_di_dl(tabelas, args.dir / f"fig_dI_dL_{args.sae}.png", args.sae, principal)
    print(f"salvo em {out} (e .pdf)")
    if args.sae == "16k" and args.paper_dir is not None:
        # figura 2 do paper: cópia feita pelo script, nunca à mão (docs/paper/README.md)
        import shutil
        args.paper_dir.mkdir(parents=True, exist_ok=True)
        for ext in (".png", ".pdf"):
            shutil.copyfile(Path(out).with_suffix(ext), args.paper_dir / f"fig2_dI_dL{ext}")
        print(f"copiada para {args.paper_dir / 'fig2_dI_dL.png'} (e .pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
