"""Estatística por feature da frente C, conforme o pré-registro (`results/frente-c/preregistro.md`).

Em cada combinação (camada × posição principal), com os latentes do SAE nas 4 células do 2×2:
- por grupo: I_g (intenção), L_g (gatilho) e X_g (interação);
- por feature: d = média/desvio dos efeitos entre grupos, IC 95% de d por bootstrap sobre grupos
  e p de Wilcoxon pareado (método de Pratt para os zeros);
- na família (features ativas × camadas × posições principais, por efeito, por SAE): q de
  Benjamini–Hochberg;
- classes S_intent / S_lex / S_mixed, com teste de equivalência (IC de d dentro de ±0,2).

Depois: combinação principal (mais S_intent na descoberta), top-20, análise por domínio,
sensibilidade e pisos de poder. A confirmação (split `test`) só é aberta com `--abrir-confirmacao`,
e só na combinação principal.

    PYTHONPATH=src python -m featgrpo.interp.feature_stats --sae 16k                     # só descoberta
    PYTHONPATH=src python -m featgrpo.interp.feature_stats --sae 16k --abrir-confirmacao  # uma vez, no fim
    PYTHONPATH=src python -m featgrpo.interp.feature_stats --sae 262k                    # C4a: mesma combinação do 16k
    PYTHONPATH=src python -m featgrpo.interp.feature_stats --modo provisorio              # pares originais (diagnóstico)

Convenção de nomes dos arquivos: `L{camada}_{posição}_{sae}`, com a posição pelo ID do cache
(`inst_last` = fim_pedido, `post_last` = pre_resposta), igual a `classifiers.py` da frente D.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import wilcoxon

from featgrpo.data.phase1.validate_template import CELLS
from featgrpo.interp import cache_io
from featgrpo.interp.data import REPO

LAYERS = (9, 17, 22, 29)
MAIN_POSITIONS = ("inst_last", "post_last")
LABELS = {"inst_last": "fim_pedido", "post_last": "pre_resposta", "post": "transicao"}
DISCOVERY = ("train", "val")
CONFIRMATION = ("test",)
OUT_DIR = REPO / "results" / "frente-c"
DOMAINS = ("bias", "cyber", "privacy", "violence")


@dataclass(frozen=True)
class Regras:
    """Limiares do pré-registro (seções 3, 4 e 5). Não mude sem registrar em results/DESVIOS.md."""
    d_min: float = 0.5            # |d| para efeito grande
    margem_eq: float = 0.2        # teste de equivalência: IC de d dentro de (-margem, +margem)
    q_max: float = 0.01           # BH
    freq_min: float = 0.05        # feature ativa em >= 5% dos itens de alguma célula (descoberta)
    n_boot: int = 1000
    seed: int = 0
    rep_d: float = 0.3            # replicação: mesmo sinal e |d_I| >= 0,3
    rep_frac: float = 0.70
    min_s_intent: int = 5
    piso_descoberta: int = 260    # seção 4, item 1
    piso_confirmacao: int = 50    # seção 4, item 2
    piso_dominio: int = 50        # seção 4, item 3
    dom_d: float = 0.3            # seção 3, item 4
    dom_geral: int = 3
    top: int = 20
    bloco: int = 16384            # features por bloco (memória com o 262k)


# ---------------------------------------------------------------- grupos e células

@dataclass
class Grupos:
    """Grupos completos do 2×2, com os índices das 4 linhas (ordem de CELLS) no índice/latentes."""
    gid: np.ndarray
    split: np.ndarray
    domain: np.ndarray
    trig_norm: np.ndarray
    rows: np.ndarray                      # (n, 4)
    excluidos: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.gid)

    def subset(self, mask: np.ndarray) -> "Grupos":
        return Grupos(self.gid[mask], self.split[mask], self.domain[mask], self.trig_norm[mask],
                      self.rows[mask], dict(self.excluidos))


def montar_grupos(index_rows: list[dict], posicao: str, items=CELLS) -> Grupos:
    """Junta as linhas do índice por grupo. Só ficam grupos com todos os `items`.

    No `inst_last` (fim_pedido), exige o mesmo `inst_last_token_id` nos itens do grupo (pré-registro,
    seção 1, regra 1). Se o campo não existir no índice, nada é excluído e isso fica registrado.
    """
    por_grupo: dict[str, dict[str, int]] = defaultdict(dict)
    for i, r in enumerate(index_rows):
        it = r.get("cell") if items == CELLS else r.get("item")
        if it in items:
            por_grupo[r["group_id"]][it] = i
    incompletos = token_diferente = 0
    sem_token_id = False
    keep: list[tuple[str, list[int]]] = []
    for g, its in por_grupo.items():
        if len(its) < len(items):
            incompletos += 1
            continue
        idx = [its[c] for c in items]
        if posicao == "inst_last":
            ids = [index_rows[i].get("inst_last_token_id") for i in idx]
            if any(t is None for t in ids):
                sem_token_id = True
            elif len(set(ids)) > 1:
                token_diferente += 1
                continue
        keep.append((g, idx))
    first = [index_rows[idx[0]] for _, idx in keep]
    total = len(por_grupo)
    excl = {"grupos_no_indice": total, "incompletos": incompletos}
    if posicao == "inst_last":
        excl["token_diferente_fim_pedido"] = token_diferente
        excl["fracao_excluida_fim_pedido"] = round(token_diferente / max(1, total - incompletos), 4)
        excl["fim_pedido_token_verificado"] = not sem_token_id
    return Grupos(
        gid=np.array([g for g, _ in keep], dtype=object),
        split=np.array([r["split"] for r in first], dtype=object),
        domain=np.array([r.get("domain") for r in first], dtype=object),
        trig_norm=np.array([r.get("trig_norm", "") for r in first], dtype=object),
        rows=np.array([idx for _, idx in keep], dtype=np.int64).reshape(-1, len(items)),
        excluidos=excl,
    )


def linhas(Z: sp.csr_matrix, g: Grupos) -> sp.csr_matrix:
    """Latentes só das linhas dos grupos, na ordem (grupo, célula)."""
    return Z[g.rows.reshape(-1)]


def bloco_celulas(sub: sp.csr_matrix, n: int, k: int, b0: int, b1: int) -> np.ndarray:
    """(k, n, b1-b0) float64: ativações das k células (ordem de CELLS) nas features [b0, b1)."""
    a = sub[:, b0:b1].toarray().astype(np.float64)
    return a.reshape(n, k, b1 - b0).transpose(1, 0, 2)


# ---------------------------------------------------------------- medidas

def efeitos(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """I, L, X por grupo; `a` é (4, n, F) na ordem (harm_trig, harm_notrig, ben_trig, ben_notrig)."""
    ht, hn, bt, bn = a
    i_g = 0.5 * (ht + hn - bt - bn)
    l_g = 0.5 * (ht + bt - hn - bn)
    x_g = 0.5 * (ht - hn - bt + bn)
    return i_g, l_g, x_g


def cohen_d(e: np.ndarray) -> np.ndarray:
    """d pareado = média/desvio (ddof=1) por coluna; 0 quando o desvio é 0."""
    m = e.mean(axis=0)
    s = e.std(axis=0, ddof=1) if e.shape[0] > 1 else np.zeros_like(m)
    return np.divide(m, s, out=np.zeros_like(m), where=s > 0)


def pesos_bootstrap(n: int, n_boot: int, seed: int) -> np.ndarray:
    """Matriz (n_boot, n) de contagens multinomiais: cada linha é um sorteio de grupos com reposição."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)


def ic_d(e: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IC 95% percentil de d por bootstrap sobre grupos, com os pesos `w` (mesmos para todas as features)."""
    n = e.shape[0]
    mu = e.mean(axis=0)
    ec = e - mu                                   # centrar evita perda de precisão em E[x²] - m²
    m_c = w @ ec / n
    v = (w @ (ec * ec) / n - m_c * m_c) * (n / (n - 1))
    s = np.sqrt(np.clip(v, 0.0, None))
    m = m_c + mu
    d = np.divide(m, s, out=np.zeros_like(m), where=s > 0)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    return lo, hi


def p_wilcoxon(e: np.ndarray) -> np.ndarray:
    """p bilateral de Wilcoxon pareado (Pratt, aproximação normal) por coluna; 1 se a coluna é toda zero."""
    p = np.ones(e.shape[1])
    nz = np.any(e != 0, axis=0)
    if nz.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = wilcoxon(e[:, nz], zero_method="pratt", method="approx", axis=0)
        p[nz] = np.nan_to_num(np.asarray(r.pvalue, dtype=float), nan=1.0)
    return p


def bh(p: np.ndarray) -> np.ndarray:
    """q de Benjamini–Hochberg."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p.copy()
    o = np.argsort(p)
    qs = p[o] * m / np.arange(1, m + 1)
    qs = np.minimum.accumulate(qs[::-1])[::-1]
    q = np.empty(m)
    q[o] = np.minimum(qs, 1.0)
    return q


# ---------------------------------------------------------------- tabela de uma combinação

def tabela_2x2(sub: sp.csr_matrix, n: int, regras: Regras) -> pd.DataFrame:
    """Uma linha por feature: frequência, d, IC, p e consistência entre grupos (descoberta)."""
    n_feat = sub.shape[1]
    w = pesos_bootstrap(n, regras.n_boot, regras.seed)
    partes = []
    for b0 in range(0, n_feat, regras.bloco):
        b1 = min(n_feat, b0 + regras.bloco)
        a = bloco_celulas(sub, n, 4, b0, b1)
        freq = (a > 0).mean(axis=1).max(axis=0)
        ativa = freq >= regras.freq_min
        nan = np.full(b1 - b0, np.nan)
        col = {k: nan.copy() for k in ("d_I", "d_L", "d_X", "d_I_ci_lo", "d_I_ci_hi", "d_L_ci_lo", "d_L_ci_hi",
                                        "p_I", "p_L", "p_X", "frac_I_pos", "frac_I_neg")}
        if ativa.any():
            i_g, l_g, x_g = (e[:, ativa] for e in efeitos(a))
            col["d_I"][ativa], col["d_L"][ativa], col["d_X"][ativa] = cohen_d(i_g), cohen_d(l_g), cohen_d(x_g)
            col["d_I_ci_lo"][ativa], col["d_I_ci_hi"][ativa] = ic_d(i_g, w)
            col["d_L_ci_lo"][ativa], col["d_L_ci_hi"][ativa] = ic_d(l_g, w)
            col["p_I"][ativa], col["p_L"][ativa], col["p_X"][ativa] = p_wilcoxon(i_g), p_wilcoxon(l_g), p_wilcoxon(x_g)
            col["frac_I_pos"][ativa], col["frac_I_neg"][ativa] = (i_g > 0).mean(axis=0), (i_g < 0).mean(axis=0)
        partes.append(pd.DataFrame({"feature": np.arange(b0, b1), "freq_ativacao": freq, "ativa": ativa, **col}))
    return pd.concat(partes, ignore_index=True)


def bh_familia(tabelas: dict, efeitos_=("I", "L", "X")) -> None:
    """Aplica BH conjunto sobre as features ativas de todas as combinações (uma família por efeito)."""
    for ef in efeitos_:
        pedacos = []
        for k, df in tabelas.items():
            idx = np.flatnonzero(df["ativa"].to_numpy())
            pedacos.append((k, idx, df[f"p_{ef}"].to_numpy()[idx]))
        q = bh(np.concatenate([p for _, _, p in pedacos])) if pedacos else np.array([])
        i = 0
        for k, idx, p in pedacos:
            col = np.full(len(tabelas[k]), np.nan)
            col[idx] = q[i:i + len(p)]
            i += len(p)
            tabelas[k][f"q_{ef}"] = col


def classificar(df: pd.DataFrame, regras: Regras) -> pd.DataFrame:
    """Classes do pré-registro (seção 5) + a regra antiga, para a análise de sensibilidade (seção 4, item 4)."""
    m = regras.margem_eq
    eq_l = (df["d_L_ci_lo"] > -m) & (df["d_L_ci_hi"] < m)
    eq_i = (df["d_I_ci_lo"] > -m) & (df["d_I_ci_hi"] < m)
    big_i, big_l = df["d_I"].abs() >= regras.d_min, df["d_L"].abs() >= regras.d_min
    sig_i, sig_l = df["q_I"] < regras.q_max, df["q_L"] < regras.q_max
    ativa = df["ativa"].astype(bool)
    df["classe"] = np.select(
        [~ativa, big_i & sig_i & eq_l, big_l & sig_l & eq_i, big_i & big_l & sig_i & sig_l],
        ["rara", "S_intent", "S_lex", "S_mixed"], default="irrelevante")
    df["S_intent_regra_antiga"] = ativa & big_i & sig_i & (df["d_L"].abs() < m)
    return df


# ---------------------------------------------------------------- escolhas na descoberta

def resumo_combinacoes(tabelas: dict, n_desc: dict, regras: Regras) -> pd.DataFrame:
    linhas_ = []
    for (layer, pos), df in tabelas.items():
        si = df[df["classe"] == "S_intent"]
        n = n_desc[(layer, pos)]
        linhas_.append({
            "camada": layer, "posicao": pos, "rotulo": LABELS.get(pos, pos), "n_grupos_descoberta": n,
            "poder_ok": n >= regras.piso_descoberta,
            "S_intent": len(si), "S_lex": int((df["classe"] == "S_lex").sum()),
            "S_mixed": int((df["classe"] == "S_mixed").sum()),
            "S_intent_regra_antiga": int(df["S_intent_regra_antiga"].sum()),
            "features_ativas": int(df["ativa"].sum()),
            "mediana_abs_d_I": float(si["d_I"].abs().median()) if len(si) else 0.0,
        })
    return pd.DataFrame(linhas_)


def escolher_principal(resumo: pd.DataFrame) -> tuple[int, str] | None:
    """Mais S_intent na descoberta; empate: maior mediana de |d_I|. Só combinações acima do piso de poder.
    Um empate também nesses dois (improvável) cai na menor camada e na ordem de MAIN_POSITIONS."""
    ok = resumo[resumo["poder_ok"]].copy()
    if ok.empty:
        return None
    ok["_ordem_pos"] = ok["posicao"].map({p: i for i, p in enumerate(MAIN_POSITIONS)}).fillna(99)
    best = ok.sort_values(["S_intent", "mediana_abs_d_I", "camada", "_ordem_pos"],
                          ascending=[False, False, True, True]).iloc[0]
    return int(best["camada"]), str(best["posicao"])


def top_k(df: pd.DataFrame, classe: str, regras: Regras) -> pd.DataFrame:
    col = "d_I" if classe == "S_intent" else "d_L"
    sel = df[df["classe"] == classe]
    return sel.reindex(sel[col].abs().sort_values(ascending=False).index).head(regras.top)


def por_dominio(sub: sp.csr_matrix, g: Grupos, feats: np.ndarray, d_global: np.ndarray, regras: Regras) -> pd.DataFrame:
    """Seção 3, item 4: d_I por domínio das S_intent e em quantos domínios o sinal se mantém."""
    if len(feats) == 0:
        return pd.DataFrame(columns=["feature"])
    a = sub[:, feats].toarray().astype(np.float64).reshape(len(g), 4, len(feats)).transpose(1, 0, 2)
    i_g = efeitos(a)[0]
    out = {"feature": feats}
    consist = np.zeros(len(feats), dtype=int)
    avaliados = 0
    for dom in DOMAINS:
        m = g.domain == dom
        n = int(m.sum())
        out[f"n_{dom}"] = np.full(len(feats), n)
        if n < regras.piso_dominio:
            out[f"d_I_{dom}"] = np.full(len(feats), np.nan)       # "dados insuficientes"
            continue
        avaliados += 1
        d = cohen_d(i_g[m])
        out[f"d_I_{dom}"] = d
        consist += ((np.sign(d) == np.sign(d_global)) & (np.abs(d) >= regras.dom_d)).astype(int)
    out["dominios_avaliados"] = np.full(len(feats), avaliados)
    out["dominios_consistentes"] = consist
    out["geral"] = consist >= regras.dom_geral
    return pd.DataFrame(out)


# ---------------------------------------------------------------- confirmação

def replicar(sub_conf: sp.csr_matrix, g_conf: Grupos, feats: np.ndarray, d_i_desc: np.ndarray,
             train_trigs: set[str], regras: Regras) -> tuple[dict, pd.DataFrame]:
    """C1 na confirmação: fração das features com o mesmo sinal e |d_I| >= 0,3 (gatilho inédito,
    ou a confirmação inteira se o inédito tiver menos grupos que o piso)."""
    a = sub_conf[:, feats].toarray().astype(np.float64).reshape(len(g_conf), 4, len(feats)).transpose(1, 0, 2)
    i_g, l_g, _ = efeitos(a)
    visto = np.array([t in train_trigs for t in g_conf.trig_norm], dtype=bool)
    cols = {"feature": feats}
    for nome, m in (("inedito", ~visto), ("visto", visto), ("todos", np.ones_like(visto))):
        ok = m.sum() > 1
        cols[f"d_I_test_{nome}"] = cohen_d(i_g[m]) if ok else np.full(len(feats), np.nan)
        cols[f"d_L_test_{nome}"] = cohen_d(l_g[m]) if ok else np.full(len(feats), np.nan)
    df = pd.DataFrame(cols)
    n_ined = int((~visto).sum())
    recorte = "inedito" if n_ined >= regras.piso_confirmacao else "todos"
    d_rep = df[f"d_I_test_{recorte}"].to_numpy()
    rep = (np.sign(d_rep) == np.sign(d_i_desc)) & (np.abs(d_rep) >= regras.rep_d)
    resumo = {
        "n_grupos_confirmacao": len(g_conf), "n_gatilho_inedito": n_ined, "n_gatilho_visto": int(visto.sum()),
        "recorte_usado": recorte, "baixo_poder": recorte != "inedito",
        "n_features": len(feats), "n_replicam": int(rep.sum()),
        "fracao_replica": float(rep.mean()) if len(feats) else None,
    }
    return resumo, df


def veredito_c1(resumo: pd.DataFrame, principal, rep: dict | None, regras: Regras) -> dict:
    if principal is None:
        return {"status": "inconclusivo", "motivo": f"nenhuma combinação com >= {regras.piso_descoberta} grupos na descoberta"}
    row = resumo[(resumo["camada"] == principal[0]) & (resumo["posicao"] == principal[1])].iloc[0]
    if not bool(row["poder_ok"]):
        return {"status": "inconclusivo", "motivo": f"{int(row['n_grupos_descoberta'])} grupos na descoberta "
                                                    f"(< {regras.piso_descoberta})"}
    n_si = int(row["S_intent"])
    if n_si < regras.min_s_intent:
        return {"status": "nao_atende", "motivo": f"{n_si} S_intent na combinação principal (< {regras.min_s_intent})"}
    if rep is None:
        return {"status": "aguardando_confirmacao", "S_intent": n_si}
    atende = rep["fracao_replica"] >= regras.rep_frac
    return {"status": "atende" if atende else "nao_atende", "S_intent": n_si,
            "fracao_replica": rep["fracao_replica"], "recorte": rep["recorte_usado"], "baixo_poder": rep["baixo_poder"]}


# ---------------------------------------------------------------- execução

def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return None


def tag(layer: int, pos: str, sae: str) -> str:
    return f"L{layer}_{pos}_{sae}"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _feats_json(df: pd.DataFrame, col: str) -> list[dict]:
    keep = ["feature", "d_I", "d_L", "d_X", "d_I_ci_lo", "d_I_ci_hi", "d_L_ci_lo", "d_L_ci_hi", "q_I", "q_L", "freq_ativacao"]
    d = df.reindex(df[col].abs().sort_values(ascending=False).index)[keep]
    return [{k: (None if pd.isna(v) else (int(v) if k == "feature" else round(float(v), 5))) for k, v in r.items()}
            for r in d.to_dict("records")]


def run_2x2(root: Path, sae: str, camadas, posicoes, out_dir: Path, abrir_confirmacao: bool,
            regras: Regras, conjunto: str = "cells2x2", principal_forcada: tuple[int, str] | None = None) -> dict:
    rows = cache_io.load_index(root, conjunto)
    train_trigs = {r["trig_norm"] for r in rows if r.get("split") == "train"}
    tabelas, n_desc, excluidos, grupos = {}, {}, {}, {}
    for layer in camadas:
        for pos in posicoes:
            Z = cache_io.load_latents(root, conjunto, layer, pos, sae)
            g = montar_grupos(rows, pos)
            desc = g.subset(np.isin(g.split, DISCOVERY))
            tabelas[(layer, pos)] = tabela_2x2(linhas(Z, desc), len(desc), regras)
            n_desc[(layer, pos)], excluidos[(layer, pos)], grupos[(layer, pos)] = len(desc), g.excluidos, g
    bh_familia(tabelas)
    for df in tabelas.values():
        classificar(df, regras)
    resumo = resumo_combinacoes(tabelas, n_desc, regras)
    principal = principal_forcada or escolher_principal(resumo)

    dominio_df, rep, rep_df = None, None, None
    if principal is not None and principal in tabelas:
        df_p, g_p = tabelas[principal], grupos[principal]
        Z = cache_io.load_latents(root, conjunto, principal[0], principal[1], sae)
        desc = g_p.subset(np.isin(g_p.split, DISCOVERY))
        s_int = df_p[df_p["classe"] == "S_intent"]
        dominio_df = por_dominio(linhas(Z, desc), desc, s_int["feature"].to_numpy(), s_int["d_I"].to_numpy(), regras)
        if abrir_confirmacao:
            conf = g_p.subset(np.isin(g_p.split, CONFIRMATION))
            sub_conf = linhas(Z, conf)
            sel = df_p[df_p["classe"].isin(["S_intent", "S_lex"])]
            _, rep_df = replicar(sub_conf, conf, sel["feature"].to_numpy(), sel["d_I"].to_numpy(), train_trigs, regras)
            rep, _ = replicar(sub_conf, conf, s_int["feature"].to_numpy(), s_int["d_I"].to_numpy(), train_trigs, regras)
    c1 = veredito_c1(resumo, principal, rep, regras) if principal in tabelas or principal is None else None

    # ---- saídas
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "listas").mkdir(exist_ok=True)
    for (layer, pos), df in tabelas.items():
        t = tag(layer, pos, sae)
        out = df[df["ativa"]].copy()
        if (layer, pos) == principal and rep_df is not None:
            out = out.merge(rep_df, on="feature", how="left")
        out.to_csv(out_dir / f"feature_classes_{t}.csv", index=False, float_format="%.5g")
        for cls in ("S_intent", "S_lex"):
            col = "d_I" if cls == "S_intent" else "d_L"
            sel = df[df["classe"] == cls]
            payload = {
                "features": [int(f) for f in sel.reindex(sel[col].abs().sort_values(ascending=False).index)["feature"]],
                "classe": cls, "camada": layer, "posicao": pos, "rotulo": LABELS.get(pos, pos), "sae": sae,
                "principal": (layer, pos) == principal, "ordenadas_por": f"|{col}| decrescente",
                "detalhe": _feats_json(sel, col),
            }
            (out_dir / "listas" / f"{cls}_{t}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")

    res = {
        "sae": sae, "conjunto": conjunto,
        "combinacoes_descoberta": resumo.to_dict("records"),
        "grupos_excluidos": {tag(k[0], k[1], sae): v for k, v in excluidos.items()},
        "principal": None if principal is None else {"camada": principal[0], "posicao": principal[1],
                                                      "rotulo": LABELS.get(principal[1], principal[1]),
                                                      "forcada_pelo_16k": principal_forcada is not None},
        "C1": c1,
        "confirmacao_aberta": bool(abrir_confirmacao),
        "replicacao_S_intent": rep,
    }
    if principal is not None and principal in tabelas:
        df_p = tabelas[principal]
        res["top20"] = {cls: _feats_json(top_k(df_p, cls, regras), "d_I" if cls == "S_intent" else "d_L")
                        for cls in ("S_intent", "S_lex")}
        res["top20_sinais_S_intent"] = {
            "positivo": int((top_k(df_p, "S_intent", regras)["d_I"] > 0).sum()),
            "negativo": int((top_k(df_p, "S_intent", regras)["d_I"] < 0).sum())}
        if dominio_df is not None and len(dominio_df):
            res["por_dominio"] = {
                "S_intent_gerais": int(dominio_df["geral"].sum()),
                "S_intent_de_um_dominio_ou_menos": int((dominio_df["dominios_consistentes"] <= 1).sum()),
                "dominios_avaliados": int(dominio_df["dominios_avaliados"].iloc[0]),
                "features": dominio_df.to_dict("records"),
            }
    return res


def run_provisorio(root: Path, sae: str, camadas, posicoes, out_dir: Path, regras: Regras) -> dict:
    """Pares originais (diagnóstico, fora do veredito): intenção = a(Q⁺) − a(Q⁰), só na descoberta."""
    rows = cache_io.load_index(root, "pairs")
    tabelas, n_desc = {}, {}
    for layer in camadas:
        for pos in posicoes:
            Z = cache_io.load_latents(root, "pairs", layer, pos, sae)
            g = montar_grupos(rows, pos, items=("q_harm", "q_border"))
            desc = g.subset(np.isin(g.split, DISCOVERY))
            n = len(desc)
            sub = linhas(Z, desc)
            w = pesos_bootstrap(n, regras.n_boot, regras.seed)
            partes = []
            for b0 in range(0, sub.shape[1], regras.bloco):
                b1 = min(sub.shape[1], b0 + regras.bloco)
                a = bloco_celulas(sub, n, 2, b0, b1)
                freq = (a > 0).mean(axis=1).max(axis=0)
                ativa = freq >= regras.freq_min
                d, lo, hi, p = (np.full(b1 - b0, np.nan) for _ in range(4))
                if ativa.any():
                    e = (a[0] - a[1])[:, ativa]
                    d[ativa], (lo[ativa], hi[ativa]), p[ativa] = cohen_d(e), ic_d(e, w), p_wilcoxon(e)
                partes.append(pd.DataFrame({"feature": np.arange(b0, b1), "freq_ativacao": freq, "ativa": ativa,
                                            "d_int": d, "d_int_ci_lo": lo, "d_int_ci_hi": hi, "p_int": p}))
            tabelas[(layer, pos)] = pd.concat(partes, ignore_index=True)
            n_desc[(layer, pos)] = n
    bh_familia(tabelas, efeitos_=("int",))
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"aviso": "Pares originais: diagnóstico, fora do veredito. Só o estilo já separa Q⁺ de Q⁰ com AUROC 0,88.",
           "combinacoes": []}
    for (layer, pos), df in tabelas.items():
        t = tag(layer, pos, sae)
        df[df["ativa"]].to_csv(out_dir / f"provisorio_{t}.csv", index=False, float_format="%.5g")
        sig = df[(df["q_int"] < regras.q_max) & (df["d_int"].abs() >= regras.d_min)]
        top = lambda s: [int(f) for f in s.reindex(s["d_int"].abs().sort_values(ascending=False).index)["feature"].head(regras.top)]
        res["combinacoes"].append({"camada": layer, "posicao": pos, "rotulo": LABELS.get(pos, pos),
                                   "n_grupos_descoberta": n_desc[(layer, pos)], "n_sig_d05": len(sig),
                                   "top20_mais_ativas_em_Q_harm": top(sig[sig["d_int"] > 0]),
                                   "top20_mais_ativas_em_Q_border": top(sig[sig["d_int"] < 0])})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Estatística por feature da frente C (ver docstring).")
    ap.add_argument("--root", type=Path, default=cache_io.DEFAULT_ROOT)
    ap.add_argument("--modo", choices=["2x2", "provisorio"], default="2x2")
    ap.add_argument("--sae", default="16k")
    ap.add_argument("--camadas", type=int, nargs="+", default=list(LAYERS))
    ap.add_argument("--posicoes", nargs="+", default=list(MAIN_POSITIONS))
    ap.add_argument("--abrir-confirmacao", action="store_true",
                    help="abre o split test na combinação principal (uma vez, no fim)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    regras = Regras()

    meta = {"script": "src/featgrpo/interp/feature_stats.py", "args": {k: str(v) for k, v in vars(args).items()},
            "regras": asdict(regras), "git_commit": git_commit(),
            "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    manifest = Path(args.root) / "MANIFEST.json"
    if manifest.exists():
        meta["cache_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))

    if args.modo == "provisorio":
        res = run_provisorio(args.root, args.sae, args.camadas, args.posicoes, args.out_dir, regras)
        path = args.out_dir / f"provisorio_{args.sae}.json"
        path.write_text(json.dumps({**meta, **res}, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        print(f"salvo em {path}")
        return 0

    summary_path = args.out_dir / "c1_c3.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    forcada = None
    if args.sae != "16k":
        p16 = (summary.get("16k") or {}).get("principal")
        if not p16:
            raise SystemExit("C4a usa a combinação principal do 16k: rode o 16k antes (pré-registro, seção 5).")
        forcada = (int(p16["camada"]), p16["posicao"])
    res = run_2x2(args.root, args.sae, args.camadas, args.posicoes, args.out_dir, args.abrir_confirmacao,
                  regras, principal_forcada=forcada)
    summary[args.sae] = {**meta, **res}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"salvo em {summary_path}\nprincipal: {res['principal']}\nC1: {res['C1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
