"""Figuras da frente C e as figuras de resultado do paper.

Paper (estrutura de 2026-11-22, revisada): Figura 2 = d_I × d_L na combinação principal, um painel; Figura 3 = queda
no XSTest em barras; Figura 4 = intervenção em barras; Figura 5 (apêndice) = d_I × d_L nos 8 painéis (camada ×
posição). A Figura 1 é um desenho em docs/paper/figuras/fonte/, e a 6 (probes, apêndice) vem da frente D.

Nos 8 painéis, eixo x: efeito de intenção (d_I); eixo y: efeito do gatilho (d_L). A faixa cinza é a zona de
equivalência (±0,2) e as linhas tracejadas, o limiar de efeito grande (±0,5). Se houver separação, as S_intent
formam um "braço" no eixo x e as S_lex, outro no eixo y.

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
    # a legenda só mostra as classes que aparecem em algum painel (S_lex e S_mixed saíram vazias)
    presentes = {c for t in tabelas.values() if t is not None and not t.empty for c in t["classe"].unique()}
    handles = [Line2D([], [], marker=marca, ls="", color=cor, markeredgecolor=SURFACE, markersize=6, label=rot)
               for cls, (cor, marca, rot) in CLASSES.items() if cls in presentes]
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


# ------------------------------------------------------------------ figuras do paper (estrutura de 2026-11-22)
# Uma ideia por figura, rótulos em palavras e valores escritos. Cores da paleta validada: azul = features de
# intenção, laranja = palavras, cinza = controles (sorteios, probe denso).
AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#7b8494"
AZUL_TXT, LARANJA_TXT, AZUL_SUAVE, LARANJA_SUAVE = "#1f5faf", "#b4441a", "#dce8f8", "#fbe3d8"


def _estilo_paper() -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
                         "ytick.color": INK_2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def _salvar(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


def scatter_principal(df: pd.DataFrame, out_path: Path, margem_eq: float = 0.2, d_min: float = 0.5) -> Path:
    """Figura 2 do paper: um painel só (a combinação principal), com as zonas rotuladas no próprio gráfico."""
    _estilo_paper()
    lim_x = max(1.0, float(np.ceil(df["d_I"].abs().max() * 10) / 10) + 0.2)
    lim_y = 1.2
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    # zonas: intenção (|d_I| ≥ 0,5 e palavra perto de zero) e palavra (o simétrico)
    for s in (-1, 1):
        ax.fill_between([s * d_min, s * lim_x], -margem_eq, margem_eq, color=AZUL_SUAVE, lw=0, zorder=0)
        ax.fill_between([-margem_eq, margem_eq], s * d_min, s * lim_y, color=LARANJA_SUAVE, lw=0, zorder=0)
    ax.axhline(0, color=GRID, lw=0.8, zorder=1)
    ax.axvline(0, color=GRID, lw=0.8, zorder=1)
    si = df["classe"] == "S_intent"
    ax.scatter(df.loc[~si, "d_I"], df.loc[~si, "d_L"], s=14, color=CINZA, alpha=0.65, lw=0, zorder=2,
               label=f"demais features ativas ({int((~si).sum())})")
    ax.scatter(df.loc[si, "d_I"], df.loc[si, "d_L"], s=38, color=AZUL, edgecolors="white", linewidths=0.7, zorder=3,
               label=f"features de intenção ({int(si.sum())})")
    ax.text(lim_x - 0.08, -margem_eq - 0.06, "região das features de intenção", ha="right", va="top", fontsize=8.5,
            color=AZUL_TXT)
    ax.text(margem_eq + 0.1, (d_min + lim_y) / 2, "região das features da palavra\n(nenhuma feature aqui)", ha="left",
            va="center", fontsize=8.5, color=LARANJA_TXT)
    ax.text(-lim_x + 0.08, -lim_y + 0.06, "← mais ativas nos pedidos benignos", ha="left", va="bottom", fontsize=8,
            color=INK_2)
    ax.text(lim_x - 0.08, -lim_y + 0.06, "mais ativas nos pedidos nocivos →", ha="right", va="bottom", fontsize=8,
            color=INK_2)
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_y, lim_y)
    ax.set_xlabel("quanto a feature reage à intenção (d)")
    ax.set_ylabel("quanto reage à palavra-gatilho (d)")
    ax.xaxis.set_major_formatter(FuncFormatter(_virgula))
    ax.yaxis.set_major_formatter(FuncFormatter(_virgula))
    ax.tick_params(length=0)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, handletextpad=0.3)
    return _salvar(fig, out_path)


def barras_xstest(ctl: dict, out_path: Path, sae: str = "16k") -> Path:
    """Figura 3 do paper: quanto a AUROC cai do nosso conjunto para o XSTest (results/frente-d/c2b_controle_aleatorio.json)."""
    _estilo_paper()
    tfidf = float(ctl[sae]["S_intent"]["queda_tfidf"])
    sorteios = [float(x) for x in ctl[sae]["aleatorio_20"]["queda_features"]]
    linhas = [("palavras (TF-IDF)", tfidf, LARANJA),
              ("48 features de intenção", float(ctl[sae]["S_intent"]["queda_features"]), AZUL),
              (f"48 features sorteadas\n(mediana de {len(sorteios)})", float(np.median(sorteios)), CINZA),
              ("probe denso\n(vetor inteiro)", float(ctl["denso"]["queda_features"]), CINZA)]
    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ys = np.arange(len(linhas))[::-1]
    # só as barras (pedido do Frederico, 22/11): a faixa dos sorteios vai na legenda do paper
    for y, (rot, v, cor) in zip(ys, linhas):
        ax.barh(y, v, height=0.56, color=cor, zorder=2)
        ax.text(v + 0.004, y, _virgula(round(v, 2)), va="center", fontsize=9.5, color=INK)
    ax.set_yticks(ys, [r for r, _, _ in linhas])
    ax.set_xlim(0, tfidf * 1.15)
    ax.xaxis.set_major_formatter(FuncFormatter(_virgula))
    ax.set_xlabel("queda da AUROC do nosso conjunto para o XSTest (menor é melhor)")
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    return _salvar(fig, out_path)


def barras_c5(c5: dict, out_path: Path) -> Path:
    """Figura 4 do paper: variação da recusa ao desligar as features (results/frente-b/c5_ablacao.json)."""
    _estilo_paper()
    grupos = [("pedidos nocivos", c5["s_intent"]["harm (test)"]),
              ("pedidos benignos que\no modelo recusava", c5["s_intent"]["hard_benign"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    w = 0.34
    for i, (_, r) in enumerate(grupos):
        v = 100 * float(r["dif_vs_base"])
        lo, hi = (100 * float(x) for x in r["ic95"])
        ax.bar(i - w / 2 - 0.02, v, width=w, color=AZUL, zorder=2)
        ax.errorbar(i - w / 2 - 0.02, v, yerr=[[v - lo], [hi - v]], color=INK, lw=1, capsize=3, zorder=3)
        ax.text(i - w / 2 - 0.02, lo - 0.6, _pts(v), ha="center", va="top", fontsize=9.5, color=INK)
        al = [100 * float(x) for x in r["dif_aleatorias"]]
        m = float(np.mean(al))
        ax.bar(i + w / 2 + 0.02, m, width=w, color=CINZA, zorder=2)   # só a barra; a faixa vai na legenda do paper
        ax.text(i + w / 2 + 0.02, max(m, 0) + 0.6, _pts(m), ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.axhline(0, color=INK_2, lw=0.8, zorder=1)
    ax.set_xticks(range(len(grupos)), [g for g, _ in grupos])
    ax.set_ylabel("variação na recusa\n(pontos percentuais)")
    ax.yaxis.set_major_formatter(FuncFormatter(_virgula))
    ax.tick_params(length=0)
    ax.set_xlim(-0.6, len(grupos) - 0.4)
    ymax = max(100 * max(float(np.mean(r["dif_aleatorias"])) for _, r in grupos), 0) + 3
    ymin = min(100 * float(r["ic95"][0]) for _, r in grupos) - 3
    ax.set_ylim(ymin, ymax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=AZUL, label="desligar as 48 features de intenção"),
               plt.Rectangle((0, 0), 1, 1, color=CINZA, label="desligar 48 features sorteadas (média de 5 sorteios)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False, fontsize=8.5)
    return _salvar(fig, out_path)


def _pts(v: float) -> str:
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):.1f}".replace(".", ",")


def figuras_paper(dir_c: Path, paper_dir: Path, sae: str = "16k") -> list[Path]:
    """Gera as figuras 2, 3 e 4 do paper e a 5 (8 painéis, apêndice), e copia para docs/paper/figuras."""
    import shutil
    raiz = dir_c.parents[1]
    resumo = json.loads((dir_c / "c1_c3.json").read_text(encoding="utf-8"))[sae]["principal"]
    camada, pos = int(resumo["camada"]), resumo["posicao"]
    df = pd.read_csv(dir_c / f"feature_classes_L{camada}_{pos}_{sae}.csv")
    ctl = json.loads((raiz / "results" / "frente-d" / "c2b_controle_aleatorio.json").read_text(encoding="utf-8"))
    c5 = json.loads((raiz / "results" / "frente-b" / "c5_ablacao.json").read_text(encoding="utf-8"))
    feitas = {
        "fig2_dI_dL": scatter_principal(df, dir_c / f"fig_dI_dL_principal_{sae}.png"),
        "fig3_xstest": barras_xstest(ctl, dir_c / f"fig_xstest_{sae}.png", sae),
        "fig4_c5": barras_c5(c5, dir_c / "fig_c5.png"),
        "fig5_dI_dL_paineis": dir_c / f"fig_dI_dL_{sae}.png",   # a grade de 8 painéis, feita no main
    }
    paper_dir.mkdir(parents=True, exist_ok=True)
    copiadas = []
    for nome, origem in feitas.items():
        for ext in (".png", ".pdf"):
            if Path(origem).with_suffix(ext).exists():
                shutil.copyfile(Path(origem).with_suffix(ext), paper_dir / f"{nome}{ext}")
                copiadas.append(paper_dir / f"{nome}{ext}")
    return copiadas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Figura d_I × d_L da frente C.")
    ap.add_argument("--sae", default="16k")
    ap.add_argument("--dir", type=Path, default=OUT_DIR)
    ap.add_argument("--paper-dir", type=Path, default=OUT_DIR.parents[1] / "docs" / "paper" / "figuras",
                    help="onde copiar as figuras do paper (só no 16k)")
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
        # figuras do paper: cópia feita pelo script, nunca à mão (docs/paper/README.md)
        for p in figuras_paper(args.dir, args.paper_dir, args.sae):
            print(f"copiada para {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
