"""Corpus do SAE próprio (A5): chat genérico + pedidos de segurança, sem nada do nosso dataset.

    python -m featgrpo.interp.prep_corpus --out /raid/.../sae_corpus

Saídas: train.jsonl e eval.jsonl, uma conversa por linha ({"messages": [...], "src": ...}).
  - HuggingFaceH4/ultrachat_200k (train_sft → treino; test_sft → avaliação)
  - PKU-SafeRLHF (pedido + resposta), excluindo os Q⁺ de todos os grupos da Fase 0
  - OR-Bench-80k (só pedido), excluindo os Q⁰ de todos os grupos da Fase 0
Ordem embaralhada com seed 0. Executar com o venv do driver (tem `datasets`).
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=str(REPO / "data" / "phase0_raw"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "train.jsonl").exists():
        print("corpus já existe"); return
    from datasets import load_dataset

    excl = set()
    for v in ("v5b", "v5"):
        with gzip.open(REPO / "datasets" / "phase0" / v / "pool_groups.jsonl.gz", "rt") as f:
            for l in f:
                g = json.loads(l)
                excl.add(norm(g["q_harm"])); excl.add(norm(g["q_border"]["text"]))
    print(f"excluídos: {len(excl)} pedidos")

    rows, ev = [], []
    uc = load_dataset("HuggingFaceH4/ultrachat_200k")
    for r in uc["train_sft"]:
        rows.append({"messages": [{"role": m["role"], "content": m["content"]} for m in r["messages"]], "src": "ultrachat"})
    for i, r in enumerate(uc["test_sft"]):
        if i >= 3000:
            break
        ev.append({"messages": [{"role": m["role"], "content": m["content"]} for m in r["messages"]], "src": "ultrachat"})
    n_uc = len(rows)
    raw = Path(a.raw)
    n_pku = 0
    with open(raw / "pku.jsonl") as f:
        for l in f:
            r = json.loads(l)
            if norm(r["prompt"]) in excl:
                continue
            for k in ("response_0", "response_1"):
                if r.get(k):
                    rows.append({"messages": [{"role": "user", "content": r["prompt"]},
                                              {"role": "assistant", "content": r[k]}], "src": "pku"}); n_pku += 1
    n_or = 0
    with open(raw / "orbench_80k.jsonl") as f:
        for l in f:
            r = json.loads(l)
            if norm(r["prompt"]) in excl:
                continue
            rows.append({"messages": [{"role": "user", "content": r["prompt"]}], "src": "orbench80k"}); n_or += 1
    random.Random(0).shuffle(rows)
    with open(out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "eval.jsonl", "w") as f:
        for r in ev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"ultrachat": n_uc, "pku": n_pku, "orbench80k": n_or, "eval": len(ev), "excluded_prompts": len(excl)}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(rep)


if __name__ == "__main__":
    main()
