"""P0 — download das fontes (§4) e conjunto de exclusão E (hash + embedding).

    python -m featgrpo.data.phase0.p0_download --fetch            # só baixa (CPU, sem GPU)
    python -m featgrpo.data.phase0.p0_download --embed-url URL    # E + embeddings (precisa do embedder)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .common import LLM, data_dir, load_config, raw_dir, log, normalize, read_jsonl, sha256, write_json, write_jsonl

TEXT_FIELDS = ("prompt", "question", "goal", "behavior", "Behavior", "forbidden_prompt")


def _ds(name, config=None, split=None):
    from datasets import load_dataset
    return load_dataset(name, config, split=split, token=os.environ.get("HF_TOKEN"))


def _text(row) -> str | None:
    for f in TEXT_FIELDS:
        if row.get(f):
            return row[f]
    return None


def fetch(raw: Path) -> dict:
    """Baixa tudo para raw/*.jsonl (idempotente)."""
    raw.mkdir(parents=True, exist_ok=True)
    counts = {}

    def save(key, rows):
        write_jsonl(raw / f"{key}.jsonl", rows)
        counts[key] = len(rows)
        log(f"P0 {key}: {len(rows)}")

    def need(key):
        p = raw / f"{key}.jsonl"
        if p.exists():
            counts[key] = sum(1 for _ in open(p))
            return False
        return True

    if need("pku"):
        save("pku", [dict(r) for r in _ds("PKU-Alignment/PKU-SafeRLHF", "default", "train")])
    for cfgname, key in [("or-bench-80k", "orbench_80k"), ("or-bench-hard-1k", "orbench_hard1k"),
                         ("or-bench-toxic", "orbench_toxic")]:
        if need(key):
            save(key, [{"prompt": r["prompt"], "category": r.get("category")} for r in _ds("bench-llm/or-bench", cfgname, "train")])
    if need("xstest"):
        d = _ds("walledai/XSTest")
        sp = list(d.keys())[0]
        save("xstest", [dict(r) for r in d[sp]])
    if need("xstest_v2_calib"):
        rows = []
        for sp in load_config()["calibration"]["splits"]:
            for r in _ds("natolambert/xstest-v2-copy", None, sp):
                rows.append(dict(r) | {"split": sp})
        save("xstest_v2_calib", rows)
    if need("falsereject_test"):
        save("falsereject_test", [{"prompt": r["prompt"], "category_text": r.get("category_text")}
                                  for r in _ds("AmazonScience/FalseReject", None, "test")])
    for cfgname in ("standard", "contextual", "copyright"):
        key = f"harmbench_{cfgname}"
        if need(key):
            d = _ds("walledai/HarmBench", cfgname)
            sp = list(d.keys())[0]
            save(key, [{"prompt": _text(r)} for r in d[sp]])
    if need("strongreject"):
        d = _ds("walledai/StrongREJECT")
        sp = list(d.keys())[0]
        save("strongreject", [{"prompt": _text(r)} for r in d[sp]])
    if need("donotanswer"):
        save("donotanswer", [{"prompt": r["question"]} for r in _ds("LibrAI/do-not-answer", None, "train")])
    return counts


EXCLUSION_KEYS = ["xstest", "orbench_hard1k", "orbench_toxic", "falsereject_test", "harmbench_standard",
                  "harmbench_contextual", "harmbench_copyright", "strongreject", "donotanswer"]


def embed_texts(llm: LLM, texts: list[str], label: str, batch: int = 256, instruction: str | None = None) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor
    chunks = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    pre = (instruction + " ") if instruction else ""

    def run(ch):
        return llm.embed([pre + t for t in ch])

    out = []
    with ThreadPoolExecutor(8) as ex:
        for i, e in enumerate(ex.map(run, chunks)):
            out.extend(e)
            if (i + 1) % 20 == 0 or i + 1 == len(chunks):
                log(f"embed {label} {min((i + 1) * batch, len(texts))}/{len(texts)}")
    a = np.asarray(out, dtype=np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    return a


def build_exclusion(raw: Path, ex_dir: Path, emb_llm: LLM) -> dict:
    ex_dir.mkdir(parents=True, exist_ok=True)
    texts, srcs = [], {}
    for k in EXCLUSION_KEYS:
        rows = read_jsonl(raw / f"{k}.jsonl")
        t = [r["prompt"] for r in rows if r.get("prompt")]
        srcs[k] = len(t)
        texts += t
    uniq = sorted({normalize(t): t for t in texts}.items())
    (ex_dir / "hashes.txt").write_text("\n".join(sha256(n) for n, _ in uniq) + "\n")
    if not (ex_dir / "emb.npy").exists():
        emb = embed_texts(emb_llm, [t for _, t in uniq], "exclusão")
        np.save(ex_dir / "emb.npy", emb)
    shape = list(np.load(ex_dir / "emb.npy", mmap_mode="r").shape)
    return {"sources": srcs, "exclusion_total": len(texts), "exclusion_unique": len(uniq), "emb_shape": shape}


class Exclusion:
    """Regra P0.5: excluído se hash ∈ E ou cos máx com E > limiar."""

    def __init__(self, ex_dir: Path, thr: float):
        self.hashes = set((ex_dir / "hashes.txt").read_text().split())
        self.emb = np.load(ex_dir / "emb.npy")
        self.thr = thr

    def mask(self, texts: list[str], emb: np.ndarray) -> np.ndarray:
        h = np.array([sha256(normalize(t)) in self.hashes for t in texts])
        cos = np.zeros(len(texts), dtype=np.float32)
        for i in range(0, len(texts), 2048):
            cos[i:i + 2048] = (emb[i:i + 2048] @ self.emb.T).max(axis=1)
        return h | (cos > self.thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--embed-url")
    a = ap.parse_args()
    cfg = load_config()
    d = data_dir()
    raw = raw_dir()
    counts = fetch(raw)
    if a.fetch:
        log(f"P0 fetch ok {counts}")
        return
    emb = LLM(a.embed_url, cfg["models"]["embedder"])
    emb.wait()
    rep = build_exclusion(raw, d / "exclusion", emb)
    # embeddings do pool de retrieval (OR-Bench-80k), documentos sem instrução (§P2.b)
    if not (d / "exclusion" / "orbench80k_emb.npy").exists():
        pool = [r["prompt"] for r in read_jsonl(raw / "orbench_80k.jsonl")]
        np.save(d / "exclusion" / "orbench80k_emb.npy", embed_texts(emb, pool, "or-bench-80k"))
    write_json(d / "p0_report.json", rep)
    log(f"P0 ok {rep}")


if __name__ == "__main__":
    main()
