"""Leitura padronizada do dataset para a Fase 1. Todas as frentes usam este módulo.

    from featgrpo.interp.data import load_pairs, prompt_items
    pairs = load_pairs()                      # 2.428 pares (lista de dicts)
    items = prompt_items(pairs)               # 4.856 itens: um por pedido (q_harm / q_border)

Cada item tem: item_id, group_id, item ("q_harm" | "q_border"), label (1 = nocivo, 0 = benigno),
text, trigger, split, domain, subdomain.

Regra: splits são por grupo; nunca escolha nada olhando o split "test".
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATASET = REPO / "datasets" / "phase0"
PHASE1 = REPO / "datasets" / "phase1"
SPLITS = ("train", "val", "test")


def _read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pairs(split: str | None = None, version: str = "v5b") -> list[dict]:
    """Pares Q⁺/Q⁰ com pedidos válidos (sem exigir respostas). split=None devolve todos."""
    rows = _read_jsonl(DATASET / version / "pairs_prompt_level.jsonl")
    if split is not None:
        assert split in SPLITS, split
        rows = [r for r in rows if r["split"] == split]
    return rows


def load_final(split: str, version: str = "v5b") -> list[dict]:
    """Grupos completos (com as 8 respostas rotuladas) do dataset final."""
    assert split in SPLITS, split
    return _read_jsonl(DATASET / version / f"{split}.jsonl")


def load_pool(version: str = "v5b") -> list[dict]:
    """Todos os grupos gerados, válidos ou não, com os rótulos de cada juiz."""
    return _read_jsonl(DATASET / version / "pool_groups.jsonl.gz")


def prompt_items(pairs: list[dict]) -> list[dict]:
    """Achata pares em itens de pedido (um por Q⁺ e um por Q⁰), prontos para o extrator."""
    out = []
    for p in pairs:
        for item, label, text in (("q_harm", 1, p["q_harm"]), ("q_border", 0, p["q_border"])):
            out.append({
                "item_id": f"{p['group_id']}:{item}",
                "group_id": p["group_id"],
                "item": item,
                "label": label,
                "text": text,
                "trigger": p["trigger"],
                "split": p["split"],
                "domain": p["domain"],
                "subdomain": p.get("subdomain"),
            })
    return out


def response_items(groups: list[dict]) -> list[dict]:
    """Achata grupos do final em itens de resposta (pedido + resposta) para teacher forcing."""
    out = []
    for g in groups:
        txt = lambda q: q["text"] if isinstance(q, dict) else q  # noqa: E731 (no final, Q⁺ e Q⁰ são objetos)
        q = {"harm": txt(g["q_harm"]), "border": txt(g["q_border"])}
        for r in g["responses"]:
            out.append({
                "item_id": f"{g['group_id']}:{r['id'][-2:]}",
                "group_id": g["group_id"],
                "item": r["id"][-2:],
                "q_role": r["q_role"],
                "role": r["role"],
                "prompt": q[r["q_role"]],
                "response": r["text"],
                "label_final": (r.get("labels") or {}).get("final"),
                "trigger": g["trigger"],
                "split": g["split"],
                "domain": g["domain"],
            })
    return out


def load_cells_2x2() -> list[dict]:
    """Células do 2×2 (frente A). Só existe depois da entrega A1."""
    p = PHASE1 / "cells_2x2.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} ainda não existe (entrega A1 da frente A)")
    return _read_jsonl(p)


if __name__ == "__main__":
    from collections import Counter
    pairs = load_pairs()
    items = prompt_items(pairs)
    print("pares:", len(pairs), dict(Counter(p["split"] for p in pairs)))
    print("itens de pedido:", len(items))
    resp = response_items(load_final("train"))
    print("itens de resposta (train):", len(resp), dict(Counter(r["item"] for r in resp)))
    # sanidade: nenhum grupo em dois splits
    seen = {}
    for p in pairs:
        assert seen.setdefault(p["group_id"], p["split"]) == p["split"]
    print("ok: splits por grupo consistentes")
