"""Leitura do cache de ativações da frente B (formato da seção 4 do AGENTS.md).

    artifacts/acts/gemma-3-4b-it/resid_post/L{L:02d}/{conjunto}__{posicao}.npy          # denso, float16 [N, d]
    artifacts/acts/gemma-3-4b-it/resid_post/L{L:02d}/{conjunto}__{posicao}__{sae}.npz   # latentes (scipy.sparse CSR)
    artifacts/acts/gemma-3-4b-it/index/{conjunto}.jsonl                                  # linha i ↔ linha i

O nome do arquivo de latentes (`__{sae}.npz`, salvo com `scipy.sparse.save_npz`) é a convenção
que a frente D assume. Se a frente B usar outra, ajuste só `latents_path`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from featgrpo.data.phase1.validate_template import CELLS, normalize_trigger
from featgrpo.interp.data import REPO

DEFAULT_ROOT = REPO / "artifacts" / "acts" / "gemma-3-4b-it"
_DENSE = re.compile(r"^(?P<conj>[a-z0-9_]+)__(?P<pos>[a-z_]+)\.npy$")


def dense_path(root: Path, conjunto: str, layer: int, pos: str) -> Path:
    return Path(root) / "resid_post" / f"L{layer:02d}" / f"{conjunto}__{pos}.npy"


def latents_path(root: Path, conjunto: str, layer: int, pos: str, sae: str) -> Path:
    return Path(root) / "resid_post" / f"L{layer:02d}" / f"{conjunto}__{pos}__{sae}.npz"


def load_index(root: Path, conjunto: str) -> list[dict]:
    """Linhas do índice, com `trig_norm` (gatilho normalizado) e `cell` (para o 2×2) acrescentados."""
    with open(Path(root) / "index" / f"{conjunto}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        r["trig_norm"] = normalize_trigger(r.get("trigger") or "")
        r["cell"] = r.get("cell") or (r["item"] if r.get("item") in CELLS else None)
    return rows


def load_dense(root: Path, conjunto: str, layer: int, pos: str) -> np.ndarray:
    return np.load(dense_path(root, conjunto, layer, pos)).astype(np.float32)


def load_latents(root: Path, conjunto: str, layer: int, pos: str, sae: str) -> sp.csr_matrix:
    """Latentes esparsos. Aceita dois formatos de `.npz`:

    1. `scipy.sparse.save_npz` (tem a chave `format`);
    2. CSR "à mão", como em `caio-frente-b.md` B2: `indices`, `values` (ou `data`), `indptr` e
       `shape` (ou `d_sae`, o nº de features do SAE). Sem `shape`/`d_sae` não dá para saber
       quantas features existem, e o carregamento falha com uma mensagem clara.
    """
    path = latents_path(root, conjunto, layer, pos, sae)
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        if "format" in keys:
            pass
        else:
            data = z["values"] if "values" in keys else z["data"]
            if "shape" in keys:
                shape = tuple(int(s) for s in z["shape"])
            elif "d_sae" in keys:
                shape = (len(z["indptr"]) - 1, int(z["d_sae"]))
            else:
                raise ValueError(f"{path}: CSR sem 'shape' nem 'd_sae'; inclua um dos dois (ver PEDIDOS da frente B)")
            return sp.csr_matrix((data, z["indices"], z["indptr"]), shape=shape)
    return sp.load_npz(path).tocsr()


def available(root: Path, conjunto: str) -> list[tuple[int, str]]:
    """Pares (camada, posição) com cache denso para o conjunto."""
    out = []
    base = Path(root) / "resid_post"
    if not base.exists():
        return out
    for ldir in sorted(base.glob("L*")):
        for f in ldir.glob(f"{conjunto}__*.npy"):
            m = _DENSE.match(f.name)
            if m and m["conj"] == conjunto:
                out.append((int(ldir.name[1:]), m["pos"]))
    return sorted(out)


def seen_mask(rows: list[dict], train_trigs: set[str]) -> np.ndarray:
    return np.array([r["trig_norm"] in train_trigs for r in rows])
