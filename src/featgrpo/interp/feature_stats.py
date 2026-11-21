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
SUBCONJUNTOS = ("qc_grammatical", "qc_pass_lenient", "qc_pass", "todos")
SUBCONJUNTO_PRINCIPAL = "qc_grammatical"   # DESVIOS 2026-11-21 (sugestão da frente A); os outros são robustez
# DESVIOS 2026-11-21 (Davi, PR #8; Gustavo, PR #11): o principal é qc_grammatical AND validator_pass; o filtro do
# validador vale em todos os subconjuntos menos `todos`. `validado=False` é robustez (o qc_grammatical sem ele).


def caminho_cells(path: Path | None = None) -> Path:
    from featgrpo.interp.data import PHASE1
    return Path(path) if path is not None else PHASE1 / "cells_2x2.jsonl"


def grupos_do_subconjunto(nome: str, path: Path | None = None, validado: bool = True) -> set[str] | None:
    """Grupos do 2×2 que passam no QC da frente A (`qc_grammatical`, `qc_pass_lenient`, `qc_pass`) e, com
    `validado=True`, também no validador estrutural da frente D (`validator_pass`). `todos` devolve None."""
    if nome == "todos":
        return None
    if nome not in SUBCONJUNTOS:
        raise ValueError(f"subconjunto desconhecido: {nome}")
    with open(caminho_cells(path), encoding="utf-8") as f:
        grupos = [g for g in map(json.loads, filter(str.strip, f))]
    if validado and not all("validator_pass" in g for g in grupos):
        raise SystemExit(f"{caminho_cells(path)} não tem a marca `validator_pass` (PR #11 da frente A). "
                         "Atualize o master ou passe --cells-jsonl com o arquivo anotado.")
    return {g["group_id"] for g in grupos if g.get(nome) and (g.get("validator_pass") or not validado)}


def sha256_arquivo(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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
        split=np.array([r.get("split") for r in first], dtype=object),
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


def pesos_bootstrap_por_grupo(grupo: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap por conglomerado: sorteia grupos com reposição e repete o peso do grupo em cada uma das
    suas linhas. Devolve a matriz (n_boot, n_linhas) para `ic_d`."""
    _, inv = np.unique(np.asarray(grupo, dtype=object), return_inverse=True)
    return pesos_bootstrap(int(inv.max()) + 1, n_boot, seed)[:, inv]


def ic_d(e: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IC 95% percentil de d por bootstrap, com os pesos `w` (mesmos para todas as features). Cada linha de
    `w` é uma reamostra; o total de cada linha é o tamanho da reamostra (n no bootstrap simples; varia no
    bootstrap por conglomerado)."""
    mu = e.mean(axis=0)
    ec = e - mu                                   # centrar evita perda de precisão em E[x²] - m²
    tot = w.sum(axis=1, keepdims=True)
    m_c = w @ ec / tot
    v = (w @ (ec * ec) / tot - m_c * m_c) * (tot / (tot - 1))
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


def por_categoria(sub: sp.csr_matrix, g: Grupos, feats: np.ndarray, d_global: np.ndarray, regras: Regras,
                  cat: np.ndarray, nomes, prefixo: str) -> pd.DataFrame:
    """d_I das S_intent dentro de cada categoria (domínio ou tipo de gatilho) e em quantas o sinal se mantém
    (mesmo sinal do d global e |d| >= regras.dom_d). Categorias com menos de regras.piso_dominio grupos ficam NaN."""
    if len(feats) == 0:
        return pd.DataFrame(columns=["feature"])
    a = sub[:, feats].toarray().astype(np.float64).reshape(len(g), 4, len(feats)).transpose(1, 0, 2)
    i_g = efeitos(a)[0]
    out = {"feature": feats}
    consist = np.zeros(len(feats), dtype=int)
    avaliados = 0
    for nome in nomes:
        m = cat == nome
        n = int(m.sum())
        out[f"n_{nome}"] = np.full(len(feats), n)
        if n < regras.piso_dominio:
            out[f"d_I_{nome}"] = np.full(len(feats), np.nan)       # "dados insuficientes"
            continue
        avaliados += 1
        d = cohen_d(i_g[m])
        out[f"d_I_{nome}"] = d
        consist += ((np.sign(d) == np.sign(d_global)) & (np.abs(d) >= regras.dom_d)).astype(int)
    out[f"{prefixo}_avaliados"] = np.full(len(feats), avaliados)
    out[f"{prefixo}_consistentes"] = consist
    return pd.DataFrame(out)


def por_dominio(sub: sp.csr_matrix, g: Grupos, feats: np.ndarray, d_global: np.ndarray, regras: Regras) -> pd.DataFrame:
    """Seção 3, item 4: d_I por domínio das S_intent e em quantos domínios o sinal se mantém."""
    df = por_categoria(sub, g, feats, d_global, regras, g.domain, DOMAINS, "dominios")
    if len(df):
        df["geral"] = df["dominios_consistentes"] >= regras.dom_geral
    return df


TIPOS_GATILHO = ("surface", "action", "intent_marker")


def tipos_de_gatilho(path: Path | None = None) -> dict[str, str]:
    """group_id → tipo do gatilho (`trigger_type` do arquivo do 2×2 da frente A)."""
    with open(caminho_cells(path), encoding="utf-8") as f:
        return {g["group_id"]: g.get("trigger_type") for g in map(json.loads, filter(str.strip, f))}


def por_tipo_gatilho(sub: sp.csr_matrix, g: Grupos, feats: np.ndarray, d_global: np.ndarray, regras: Regras,
                     tipos: dict[str, str]) -> pd.DataFrame:
    """Seção 5, regra 5 (secundária, descritiva): d_I das S_intent por tipo de gatilho."""
    cat = np.array([tipos.get(x) for x in g.gid], dtype=object)
    return por_categoria(sub, g, feats, d_global, regras, cat, TIPOS_GATILHO, "tipos")


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


CONTROL_ITEMS = ("trig_only", "neutral_only")
_KIND = {"trig": "trig_only", "neutral": "neutral_only", "trig_only": "trig_only", "neutral_only": "neutral_only"}


def normalizar_controles(rows: list[dict]) -> list[dict]:
    """Aceita o formato da frente A (`kind` = trig/neutral, `group_id` do 2×2, `template_id` 0–2) e o
    formato curto (`item` = trig_only/neutral_only, `group_id` do par). Devolve cópias com `item` e
    `group_id` = identificador do par de frases, e `grupo_2x2` = grupo de origem."""
    out = []
    for r in rows:
        r2 = dict(r)
        r2["item"] = _KIND.get(r.get("item")) or _KIND.get(r.get("kind")) or r.get("item")
        origem = r.get("group_id")
        if r.get("template_id") is not None:
            r2["group_id"] = f"{origem}:ctl{r['template_id']}"
        r2["grupo_2x2"] = origem
        out.append(r2)
    return out


def controle_trig_only(z_ctrl: sp.csr_matrix, rows_ctrl: list[dict], posicao: str, feats_int: np.ndarray,
                       feats_lex: np.ndarray, regras: Regras, unidade: str = "frase",
                       reamostragem: str = "grupo") -> dict:
    """C3, item 1 (pré-registro, seção 5): d de a(trig_only) − a(neutral_only), pareado por frase de controle,
    na mesma posição da combinação principal. Atende se mais da metade das S_intent tiver o IC 95% desse d
    dentro de ±0,2. Para S_lex espera-se o contrário (o controle funciona se elas reagem).

    `unidade="frase"` (pré-registrado): o d é calculado sobre os pares de frases. `unidade="grupo"`
    (alternativa, não usada): média das frases de cada grupo do 2×2 antes do d.

    `reamostragem` (só com `unidade="frase"`): `"grupo"` (DESVIOS 2026-11-21, o principal) sorteia grupos
    do 2×2 inteiros, com as suas frases, porque as 3 frases de um grupo não são independentes; `"frase"`
    (a versão original do pré-registro, mantida como sensibilidade) sorteia pares de frases.
    """
    feats_int, feats_lex = np.asarray(feats_int, dtype=int), np.asarray(feats_lex, dtype=int)
    rows_ctrl = normalizar_controles(rows_ctrl)
    g = montar_grupos(rows_ctrl, posicao, items=CONTROL_ITEMS)
    n, k = len(g), len(feats_int)
    feats = np.concatenate([feats_int, feats_lex])
    if unidade not in ("frase", "grupo"):
        raise ValueError(f"unidade desconhecida: {unidade}")
    if reamostragem not in ("frase", "grupo"):
        raise ValueError(f"reamostragem desconhecida: {reamostragem}")
    base = {"n_pares_controle": n, "excluidos": g.excluidos, "posicao": posicao, "unidade": unidade,
            "reamostragem": "grupo" if unidade == "grupo" else reamostragem}
    if n < 2 or len(feats) == 0:
        return {**base, "status": "sem dados suficientes"}
    a = linhas(z_ctrl, g)[:, feats].toarray().astype(np.float64).reshape(n, 2, len(feats)).transpose(1, 0, 2)
    e = a[0] - a[1]
    origem = np.array([rows_ctrl[i]["grupo_2x2"] for i in g.rows[:, 0]], dtype=object)
    _, inv = np.unique(origem, return_inverse=True)
    base["n_grupos_2x2"] = int(inv.max()) + 1
    if unidade == "grupo":
        soma = np.zeros((inv.max() + 1, e.shape[1]))
        np.add.at(soma, inv, e)
        e = soma / np.bincount(inv)[:, None]
        base["n_unidades"] = int(e.shape[0])
        w = pesos_bootstrap(e.shape[0], regras.n_boot, regras.seed)
    elif reamostragem == "grupo":
        w = pesos_bootstrap_por_grupo(origem, regras.n_boot, regras.seed)
    else:
        w = pesos_bootstrap(n, regras.n_boot, regras.seed)
    d = cohen_d(e)
    lo, hi = ic_d(e, w)
    m = regras.margem_eq
    sem_efeito = (lo > -m) & (hi < m)
    frac_int = float(sem_efeito[:k].mean()) if k else None
    frac_lex = float((~sem_efeito[k:]).mean()) if len(feats_lex) else None
    detalhe = [{"feature": int(f), "classe": "S_intent" if j < k else "S_lex", "d": round(float(d[j]), 4),
                "ic_lo": round(float(lo[j]), 4), "ic_hi": round(float(hi[j]), 4), "sem_efeito": bool(sem_efeito[j])}
               for j, f in enumerate(feats)]
    return {**base, "status": "ok",
            "S_intent": {"n": k, "fracao_sem_efeito": frac_int,
                         "atende": None if frac_int is None else frac_int > 0.5},
            "S_lex": {"n": int(len(feats_lex)), "fracao_que_reage": frac_lex},
            "features": detalhe}


def estilo(textos: list[str]) -> np.ndarray:
    """Os 14 atributos de estilo do classificador "só estilo" da frente D (`baselines.style_features`)."""
    from featgrpo.interp.baselines import style_features
    return style_features(textos)


def controle_estilo(i_g: np.ndarray, ds: np.ndarray, d_desc: np.ndarray, regras: Regras) -> tuple[dict, pd.DataFrame]:
    """Análise secundária (caminho c): o efeito de intenção sobrevive quando o estilo é igual?

    Para cada feature, regressão por grupo I_g = α + β·Δs_g + ε, com Δs_g = estilo(nocivo) − estilo(benigno)
    (14 atributos, divididos pelo desvio mas **sem centrar**, para que α seja o I esperado com Δs = 0, isto é,
    com estilo igual nas duas linhas). d_I ajustado = α / desvio dos resíduos. A feature "sobrevive" se o
    d_I ajustado mantém o sinal do d_I da descoberta com |d| >= regras.rep_d (o limiar da replicação).
    """
    n = i_g.shape[0]
    ds = np.asarray(ds, dtype=np.float64)
    sd = ds.std(axis=0, ddof=1)
    ok = sd > 0
    z = ds[:, ok] / sd[ok]
    x = np.column_stack([np.ones(n), z])
    beta, *_ = np.linalg.lstsq(x, i_g, rcond=None)
    resid = i_g - x @ beta
    dof = max(n - x.shape[1], 1)
    sd_res = np.sqrt((resid ** 2).sum(axis=0) / dof)
    d_adj = np.divide(beta[0], sd_res, out=np.zeros_like(beta[0]), where=sd_res > 0)
    var_i = i_g.var(axis=0)
    r2 = np.where(var_i > 0, 1 - resid.var(axis=0) / np.where(var_i > 0, var_i, 1), 0.0)
    sobrevive = (np.sign(d_adj) == np.sign(d_desc)) & (np.abs(d_adj) >= regras.rep_d)
    df = pd.DataFrame({"d_I": d_desc, "d_I_ajustado_estilo": d_adj, "r2_estilo": r2, "sobrevive_estilo": sobrevive})
    resumo = {"n_grupos": n, "atributos_de_estilo_usados": int(ok.sum()),
              "diferenca_media_de_estilo_padronizada": [round(float(v), 3) for v in (ds[:, ok].mean(0) / sd[ok])],
              "n_features": int(len(d_desc)), "n_sobrevivem": int(sobrevive.sum()),
              "fracao_sobrevive": float(sobrevive.mean()) if len(d_desc) else None,
              "r2_mediano": float(np.median(r2)) if len(d_desc) else None}
    return resumo, df


def delta_estilo_2x2(g: Grupos, textos: dict[tuple[str, str], str]) -> np.ndarray | None:
    """Δs_g = ½[(s(harm_trig) − s(ben_trig)) + (s(harm_notrig) − s(ben_notrig))], no mesmo formato de I_g."""
    s = {}
    for c in CELLS:
        tx = [textos.get((gid, c)) for gid in g.gid]
        if any(t is None for t in tx):
            return None
        s[c] = estilo(tx)
    return 0.5 * ((s["harm_trig"] - s["ben_trig"]) + (s["harm_notrig"] - s["ben_notrig"]))


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


def _efeito_i(sub: sp.csr_matrix, n: int, k_cells: int, feats: np.ndarray) -> np.ndarray:
    """I_g (2×2) ou a(Q⁺) − a(Q⁰) (pares) só para as features `feats`."""
    a = sub[:, feats].toarray().astype(np.float64).reshape(n, k_cells, len(feats)).transpose(1, 0, 2)
    return efeitos(a)[0] if k_cells == 4 else a[0] - a[1]


def _textos_2x2_padrao() -> dict:
    from featgrpo.interp.autointerp import textos_2x2
    return textos_2x2()


def run_2x2(root: Path, sae: str, camadas, posicoes, out_dir: Path, abrir_confirmacao: bool,
            regras: Regras, conjunto: str = "cells2x2", principal_forcada: tuple[int, str] | None = None,
            conjunto_controles: str = "controls", textos: dict | None = None,
            grupos_permitidos: set[str] | None = None, subconjunto: str = "todos",
            tipos_gatilho: dict[str, str] | None = None) -> dict:
    rows = cache_io.load_index(root, conjunto)
    train_trigs = {r["trig_norm"] for r in rows if r.get("split") == "train"}
    tabelas, n_desc, excluidos, grupos = {}, {}, {}, {}
    for layer in camadas:
        for pos in posicoes:
            Z = cache_io.load_latents(root, conjunto, layer, pos, sae)
            g = montar_grupos(rows, pos)
            if grupos_permitidos is not None:
                dentro = np.array([gid in grupos_permitidos for gid in g.gid], dtype=bool)
                g = g.subset(dentro)
                g.excluidos["fora_do_subconjunto"] = int((~dentro).sum())
            desc = g.subset(np.isin(g.split, DISCOVERY))
            tabelas[(layer, pos)] = tabela_2x2(linhas(Z, desc), len(desc), regras)
            n_desc[(layer, pos)], excluidos[(layer, pos)], grupos[(layer, pos)] = len(desc), g.excluidos, g
    bh_familia(tabelas)
    for df in tabelas.values():
        classificar(df, regras)
    resumo = resumo_combinacoes(tabelas, n_desc, regras)
    principal = principal_forcada or escolher_principal(resumo)

    dominio_df, tipo_df, rep, rep_df = None, None, None, None
    estilo_res = {"status": "sem combinação principal"}
    if principal is not None and principal in tabelas:
        df_p, g_p = tabelas[principal], grupos[principal]
        Z = cache_io.load_latents(root, conjunto, principal[0], principal[1], sae)
        desc = g_p.subset(np.isin(g_p.split, DISCOVERY))
        s_int = df_p[df_p["classe"] == "S_intent"]
        dominio_df = por_dominio(linhas(Z, desc), desc, s_int["feature"].to_numpy(), s_int["d_I"].to_numpy(), regras)
        if tipos_gatilho:
            tipo_df = por_tipo_gatilho(linhas(Z, desc), desc, s_int["feature"].to_numpy(), s_int["d_I"].to_numpy(),
                                       regras, tipos_gatilho)
        estilo_res = {"status": "sem textos do 2×2 (datasets/phase1/cells_2x2.jsonl)"}
        tx = textos if textos is not None else _textos_2x2_padrao()
        if len(s_int) == 0:
            estilo_res = {"status": "sem S_intent"}
        elif tx:
            ds = delta_estilo_2x2(desc, tx)
            if ds is not None:
                feats_si = s_int["feature"].to_numpy()
                resumo_e, df_e = controle_estilo(_efeito_i(linhas(Z, desc), len(desc), 4, feats_si), ds,
                                                 s_int["d_I"].to_numpy(), regras)
                estilo_res = {"status": "ok", **resumo_e,
                              "features": df_e.assign(feature=feats_si).round(4).to_dict("records")}
        if abrir_confirmacao:
            conf = g_p.subset(np.isin(g_p.split, CONFIRMATION))
            sub_conf = linhas(Z, conf)
            sel = df_p[df_p["classe"].isin(["S_intent", "S_lex"])]
            _, rep_df = replicar(sub_conf, conf, sel["feature"].to_numpy(), sel["d_I"].to_numpy(), train_trigs, regras)
            rep, _ = replicar(sub_conf, conf, s_int["feature"].to_numpy(), s_int["d_I"].to_numpy(), train_trigs, regras)
    c1 = veredito_c1(resumo, principal, rep, regras) if principal in tabelas or principal is None else None

    controle = {"status": "sem combinação principal"}
    controle_sens = None
    if principal is not None and principal in tabelas:
        controle = {"status": f"sem cache de controles ('{conjunto_controles}') para esta combinação"}
        if cache_io.latents_path(root, conjunto_controles, principal[0], principal[1], sae).exists():
            df_p = tabelas[principal]
            rows_c = cache_io.load_index(root, conjunto_controles)
            if grupos_permitidos is not None:
                rows_c = [r for r in rows_c if r.get("group_id") in grupos_permitidos]
            args_c = (cache_io.load_latents(root, conjunto_controles, principal[0], principal[1], sae),
                      rows_c, principal[1],
                      df_p.loc[df_p["classe"] == "S_intent", "feature"].to_numpy(),
                      df_p.loc[df_p["classe"] == "S_lex", "feature"].to_numpy(), regras)
            controle = controle_trig_only(*args_c, reamostragem="grupo")      # DESVIOS 2026-11-21
            sens = controle_trig_only(*args_c, reamostragem="frase")          # versão original, sensibilidade
            controle_sens = {k: v for k, v in sens.items() if k != "features"}

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
        "sae": sae, "conjunto": conjunto, "subconjunto": subconjunto,
        "combinacoes_descoberta": resumo.to_dict("records"),
        "grupos_excluidos": {tag(k[0], k[1], sae): v for k, v in excluidos.items()},
        "principal": None if principal is None else {"camada": principal[0], "posicao": principal[1],
                                                      "rotulo": LABELS.get(principal[1], principal[1]),
                                                      "forcada_pelo_16k": principal_forcada is not None},
        "C1": c1,
        "confirmacao_aberta": bool(abrir_confirmacao),
        "replicacao_S_intent": rep,
        "C3_controle_trig_only": controle,
        "C3_controle_trig_only_sensibilidade_reamostragem_frase": controle_sens,
        "C1_controle_estilo": estilo_res,
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
        if tipo_df is not None and len(tipo_df):
            res["por_tipo_gatilho"] = {
                "descritivo": True,
                "tipos": {t_: {"n_grupos_descoberta": int(tipo_df[f"n_{t_}"].iloc[0]),
                               "S_intent_com_mesmo_sinal_e_d_ge_0_3": int(((np.sign(tipo_df[f"d_I_{t_}"]) ==
                                                                            np.sign(df_p.set_index("feature").loc[tipo_df["feature"], "d_I"].to_numpy()))
                                                                           & (tipo_df[f"d_I_{t_}"].abs() >= regras.dom_d)).sum()),
                               "mediana_abs_d_I": float(tipo_df[f"d_I_{t_}"].abs().median())}
                          for t_ in TIPOS_GATILHO if tipo_df[f"d_I_{t_}"].notna().any()},
                "n_S_intent": int(len(tipo_df)),
                "features": tipo_df.to_dict("records"),
            }
    return res


def consistencia(e: np.ndarray) -> float:
    """Norma do efeito médio ÷ média das normas por grupo: 1 = todos os grupos na mesma direção; ~0 = cada
    grupo numa direção. Serve para o efeito por grupo (I_g ou L_g) no espaço denso."""
    return float(np.linalg.norm(e.mean(axis=0)) / np.linalg.norm(e, axis=1).mean())


def diagnostico_lexical(root: Path, sae: str, camadas, posicoes, conjunto: str = "cells2x2",
                        grupos_permitidos: set[str] | None = None) -> list[dict]:
    """Análise descritiva (não decide nada): por que o d entre grupos não acha S_lex. Compara, na descoberta,
    o tamanho e a consistência do efeito de intenção e do efeito do gatilho no vetor denso, e conta quantas
    features do SAE mudam de estado (ativa ↔ inativa) com a troca do gatilho em cada grupo."""
    rows = cache_io.load_index(root, conjunto)
    out = []
    for layer in camadas:
        for pos in posicoes:
            g = montar_grupos(rows, pos)
            if grupos_permitidos is not None:
                g = g.subset(np.array([x in grupos_permitidos for x in g.gid], dtype=bool))
            g = g.subset(np.isin(g.split, DISCOVERY))
            n = len(g)
            x = cache_io.load_dense(root, conjunto, layer, pos)[g.rows.reshape(-1)].astype(np.float64).reshape(n, 4, -1)
            i_g = 0.5 * (x[:, 0] + x[:, 1] - x[:, 2] - x[:, 3])     # ordem de CELLS: ht, hn, bt, bn
            l_g = 0.5 * (x[:, 0] + x[:, 2] - x[:, 1] - x[:, 3])
            z = linhas(cache_io.load_latents(root, conjunto, layer, pos, sae), g).toarray().reshape(n, 4, -1) > 0
            muda = np.concatenate([z[:, 0] != z[:, 1], z[:, 2] != z[:, 3]])   # troca do gatilho, nas duas linhas
            ativas = np.concatenate([z[:, 0] | z[:, 1], z[:, 2] | z[:, 3]]).sum(axis=1)
            # o gatilho empurra na direção da intenção? projeção de L_g no eixo da intenção média
            u = i_g.mean(axis=0) / np.linalg.norm(i_g.mean(axis=0))
            proj_l, proj_i = l_g @ u, i_g @ u
            out.append({
                "camada": layer, "posicao": pos, "rotulo": LABELS.get(pos, pos), "n_grupos_descoberta": n,
                "norma_media_I": round(float(np.linalg.norm(i_g, axis=1).mean()), 1),
                "norma_media_L": round(float(np.linalg.norm(l_g, axis=1).mean()), 1),
                "consistencia_I": round(consistencia(i_g), 3), "consistencia_L": round(consistencia(l_g), 3),
                "cos_L_medio_I_medio": round(float(l_g.mean(axis=0) @ u / np.linalg.norm(l_g.mean(axis=0))), 3),
                "L_no_eixo_da_intencao_razao": round(float(proj_l.mean() / proj_i.mean()), 4),
                "L_no_eixo_da_intencao_d": round(float(proj_l.mean() / proj_l.std(ddof=1)), 3),
                "L_no_eixo_da_intencao_frac_grupos_positivos": round(float((proj_l > 0).mean()), 3),
                "sae": sae,
                "features_que_mudam_com_o_gatilho_mediana": float(np.median(muda.sum(axis=1))),
                "features_ativas_mediana": float(np.median(ativas)),
                "maior_fracao_de_grupos_em_que_uma_feature_muda": round(float(muda.mean(axis=0).max()), 3),
            })
    return out


def _textos_pares_padrao() -> dict:
    from featgrpo.interp.data import load_pairs
    out = {}
    for p in load_pairs():
        out[(p["group_id"], "q_harm")], out[(p["group_id"], "q_border")] = p["q_harm"], p["q_border"]
    return out


def run_provisorio(root: Path, sae: str, camadas, posicoes, out_dir: Path, regras: Regras,
                   textos: dict | None = None) -> dict:
    """Pares originais (diagnóstico, fora do veredito): intenção = a(Q⁺) − a(Q⁰), só na descoberta.
    Inclui o controle de estilo (caminho c) nas features com |d| >= 0,5 e q < 0,01."""
    rows = cache_io.load_index(root, "pairs")
    tx = textos if textos is not None else _textos_pares_padrao()
    tabelas, n_desc, subs = {}, {}, {}
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
            subs[(layer, pos)] = (sub, desc)
    bh_familia(tabelas, efeitos_=("int",))
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"aviso": "Pares originais: diagnóstico, fora do veredito. Só o estilo já separa Q⁺ de Q⁰ com AUROC 0,88.",
           "combinacoes": []}
    for (layer, pos), df in tabelas.items():
        t = tag(layer, pos, sae)
        df[df["ativa"]].to_csv(out_dir / f"provisorio_{t}.csv", index=False, float_format="%.5g")
        sig = df[(df["q_int"] < regras.q_max) & (df["d_int"].abs() >= regras.d_min)]
        top = lambda s: [int(f) for f in s.reindex(s["d_int"].abs().sort_values(ascending=False).index)["feature"].head(regras.top)]
        estilo_res = {"status": "sem features com efeito grande"}
        sub, desc = subs[(layer, pos)]
        textos_h = [tx.get((gid, "q_harm")) for gid in desc.gid]
        textos_b = [tx.get((gid, "q_border")) for gid in desc.gid]
        if len(sig) and all(textos_h) and all(textos_b):
            feats = sig["feature"].to_numpy()
            resumo_e, _ = controle_estilo(_efeito_i(sub, len(desc), 2, feats), estilo(textos_h) - estilo(textos_b),
                                          sig["d_int"].to_numpy(), regras)
            estilo_res = {"status": "ok", **{k: v for k, v in resumo_e.items()
                                             if k != "diferenca_media_de_estilo_padronizada"}}
        res["combinacoes"].append({"camada": layer, "posicao": pos, "rotulo": LABELS.get(pos, pos),
                                   "n_grupos_descoberta": n_desc[(layer, pos)], "n_sig_d05": len(sig),
                                   "top20_mais_ativas_em_Q_harm": top(sig[sig["d_int"] > 0]),
                                   "top20_mais_ativas_em_Q_border": top(sig[sig["d_int"] < 0]),
                                   "controle_estilo": estilo_res})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Estatística por feature da frente C (ver docstring).")
    ap.add_argument("--root", type=Path, default=cache_io.DEFAULT_ROOT)
    ap.add_argument("--modo", choices=["2x2", "provisorio", "diagnostico-lexical"], default="2x2")
    ap.add_argument("--sae", default="16k")
    ap.add_argument("--camadas", type=int, nargs="+", default=list(LAYERS))
    ap.add_argument("--posicoes", nargs="+", default=list(MAIN_POSITIONS))
    ap.add_argument("--abrir-confirmacao", action="store_true",
                    help="abre o split test na combinação principal (uma vez, no fim)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="padrão: results/frente-c (principal) ou results/frente-c/robustez_<subconjunto>")
    ap.add_argument("--conjunto", default="cells2x2",
                    help="conjunto do 2×2 no cache (o caminho (b), em estilo comum, sai com outro nome)")
    ap.add_argument("--cells-jsonl", type=Path, default=None,
                    help="arquivo do 2×2 da frente A, para o QC e os textos do controle de estilo "
                         "(padrão: datasets/phase1/cells_2x2.jsonl; caminho (b): cells_2x2_style.jsonl)")
    ap.add_argument("--conjunto-controles", default="controls",
                    help="conjunto do cache com os controles trig_only/neutral_only (C3)")
    ap.add_argument("--subconjunto", choices=SUBCONJUNTOS, default=SUBCONJUNTO_PRINCIPAL,
                    help="grupos do 2×2 pelo QC da frente A; o principal é qc_grammatical (com o validador), "
                         "os outros são robustez")
    ap.add_argument("--sem-validador", action="store_true",
                    help="não exige validator_pass (robustez; o principal exige, DESVIOS 2026-11-21)")
    args = ap.parse_args(argv)
    regras = Regras()
    principal_run = args.subconjunto == SUBCONJUNTO_PRINCIPAL and not args.sem_validador
    if args.out_dir is None:
        nome = args.subconjunto + ("_sem_validador" if args.sem_validador else "")
        args.out_dir = OUT_DIR if (principal_run or args.modo == "provisorio") else OUT_DIR / f"robustez_{nome}"

    meta = {"script": "src/featgrpo/interp/feature_stats.py", "args": {k: str(v) for k, v in vars(args).items()},
            "regras": asdict(regras), "git_commit": git_commit(),
            "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if args.modo != "provisorio":
        cp = caminho_cells(args.cells_jsonl)
        meta["cells_jsonl"] = {"caminho": str(cp), "sha256": sha256_arquivo(cp)}
    if args.modo == "diagnostico-lexical":
        res = diagnostico_lexical(args.root, args.sae, args.camadas, args.posicoes, args.conjunto,
                                  grupos_do_subconjunto(args.subconjunto, args.cells_jsonl,
                                                        validado=not args.sem_validador))
        path = args.out_dir / f"diagnostico_lexical_{args.sae}.json"
        path.write_text(json.dumps({**meta, "combinacoes": res}, ensure_ascii=False, indent=2,
                                   default=_json_default), encoding="utf-8")
        print(f"salvo em {path}")
        return 0
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
    main_summary_path = OUT_DIR / "c1_c3.json"
    main_summary = json.loads(main_summary_path.read_text(encoding="utf-8")) if main_summary_path.exists() else {}
    forcada = None
    if args.sae != "16k" or not principal_run:
        # C4a (262k) e as robustezes de QC usam a combinação principal do 16k no subconjunto principal
        p16 = (main_summary.get("16k") or {}).get("principal")
        if not p16:
            raise SystemExit("Rode antes o 16k no subconjunto principal (qc_grammatical): C4a e as robustezes "
                             "usam a combinação principal dele (pré-registro, seção 5; DESVIOS 2026-11-21).")
        forcada = (int(p16["camada"]), p16["posicao"])
    textos = None
    if args.cells_jsonl is not None:
        from featgrpo.interp.autointerp import textos_2x2
        textos = textos_2x2(args.cells_jsonl)
    res = run_2x2(args.root, args.sae, args.camadas, args.posicoes, args.out_dir, args.abrir_confirmacao,
                  regras, conjunto=args.conjunto, principal_forcada=forcada,
                  conjunto_controles=args.conjunto_controles, textos=textos,
                  grupos_permitidos=grupos_do_subconjunto(args.subconjunto, args.cells_jsonl,
                                                          validado=not args.sem_validador),
                  tipos_gatilho=tipos_de_gatilho(args.cells_jsonl),
                  subconjunto=args.subconjunto + ("" if args.subconjunto == "todos" or args.sem_validador
                                                  else "+validator_pass"))
    summary[args.sae] = {**meta, **res}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"salvo em {summary_path}\nprincipal: {res['principal']}\nC1: {res['C1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
