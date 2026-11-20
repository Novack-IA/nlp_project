"""A-P2: tipologia dos gatilhos (surface / action / intent_marker) pelo juiz de 70B.

    python -m featgrpo.data.phase1.t1_types --judge-url URL
Saída: datasets/phase1/trigger_types.jsonl (um por gatilho único, normalizado em minúsculas).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from ..phase0.common import LLM, get_cache, pmap, read_jsonl, write_json, write_jsonl
from .common1 import JUDGE_MODEL, PAIRS, chat_template, data1, fill, judge_json, log, out_dir

TYPES = ["surface", "action", "intent_marker"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-url", required=True)
    ap.add_argument("--batch", type=int, default=20)
    a = ap.parse_args()
    out = out_dir() / "trigger_types.jsonl"
    if out.exists():
        log(f"t1: {out} já existe"); return
    pairs = read_jsonl(PAIRS)
    ex = {}
    for p in pairs:
        ex.setdefault(p["trigger"].lower(), p)
    trig = sorted(ex)
    log(f"t1: {len(trig)} gatilhos únicos")
    llm = LLM(a.judge_url, JUDGE_MODEL); llm.wait()
    cache = get_cache()
    _, utmpl = chat_template("trigger_type_v1")
    batches = [trig[i:i + a.batch] for i in range(0, len(trig), a.batch)]

    def run(b):
        items = [{"id": str(i), "trigger": t, "example_harmful": ex[t]["q_harm"], "example_benign": ex[t]["q_border"]}
                 for i, t in enumerate(b)]
        ids = [x["id"] for x in items]
        schema = {"type": "object", "properties": {"items": {"type": "array", "minItems": len(ids), "maxItems": len(ids),
                  "items": {"type": "object", "properties": {"id": {"enum": ids}, "type": {"enum": TYPES}},
                            "required": ["id", "type"]}}}, "required": ["items"]}
        ok = lambda o: {x["id"] for x in o["items"]} == set(ids) and all(x["type"] in TYPES for x in o["items"])
        o = judge_json(llm, cache, "trigger_type_v1", fill(utmpl, items=json.dumps(items, ensure_ascii=False, indent=1)),
                       schema, ok, max_tokens=2048)
        got = {x["id"]: x["type"] for x in o["items"]} if o else {}
        return [{"trigger": t, "type": got.get(str(i))} for i, t in enumerate(b)]

    rows = [r for rs in pmap(run, batches, workers=32, label="t1") for r in rs]
    n_pairs = Counter(p["trigger"].lower() for p in pairs)
    for r in rows:
        r["n_pairs"] = n_pairs[r["trigger"]]
    write_jsonl(out, rows)
    rep = {"triggers": len(rows), "types": dict(Counter(str(r["type"]) for r in rows))}
    pb = Counter()
    for r in rows:
        pb[str(r["type"])] += r["n_pairs"]
    rep["pairs_by_type"] = dict(pb)
    rep["run"] = llm.stats
    write_json(data1() / "t1_report.json", rep)
    log(f"t1 ok {rep}")


if __name__ == "__main__":
    main()
