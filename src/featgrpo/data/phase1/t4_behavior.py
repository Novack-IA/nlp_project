"""A4: comportamento natural do Gemma-3-4B-IT (sem system prompt) e rotulagem.

    python -m featgrpo.data.phase1.t4_behavior generate --url URL --set pairs|external|cells
    python -m featgrpo.data.phase1.t4_behavior label --qg-url U --wg-url U --judge-url U --set pairs|external|cells

generate → PHASE1_DATA/behavior_raw_<set>.jsonl (+ .done)
label    → PHASE1_DATA/behavior_<set>.jsonl; quando os 3 conjuntos existem, junta em
           datasets/phase1/behavior_gemma3_4b.jsonl, hcr_orr_4b.json e hard_benign_4b.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from ..phase0 import guards as G
from ..phase0.common import LLM, get_cache, pmap, read_jsonl, write_json, write_jsonl
from ..phase0.p4_label import consolidate_response, judge_text
from .common1 import (ACTIVE_GUARDS, JUDGE_MODEL, PAIRS, SUBJECT4B, chat_template, data1, fill, judge_json, log,
                      out_dir)

SETS = ["pairs", "external", "cells", "mining"]
GEN = {"temperature": 0.7, "top_p": 1.0, "seed": 0, "max_tokens": 512}
CLASSES = ["unsafe_compliance", "safe_helpful", "partial_refusal", "refusal"]


def items(which: str) -> list[dict]:
    if which == "pairs":
        out = []
        for p in read_jsonl(PAIRS):
            out.append({"item_id": f"{p['group_id']}:q_harm", "set": "pairs", "cell": "orig_harm", "label": "harmful",
                        "text": p["q_harm"], "group_id": p["group_id"], "domain": p["domain"], "split": p["split"]})
            out.append({"item_id": f"{p['group_id']}:q_border", "set": "pairs", "cell": "orig_benign",
                        "label": "benign", "text": p["q_border"], "group_id": p["group_id"], "domain": p["domain"],
                        "split": p["split"]})
        return out
    if which == "external":
        out = []
        for name in ("xstest", "orbench_hard", "falsereject"):
            for r in read_jsonl(out_dir() / "external" / f"{name}.jsonl"):
                out.append({"item_id": r["id"], "set": name, "cell": r.get("type"), "label": r["label"],
                            "text": r["text"], "group_id": None, "domain": None, "split": None})
        return out
    if which == "mining":
        return [{"item_id": r["item_id"], "set": "mining", "cell": "mined_benign", "label": "benign", "text": r["text"],
                 "group_id": r["group_id"], "domain": r["domain"], "split": r["split"]}
                for r in read_jsonl(data1() / "mining_variants.jsonl")]
    out = []
    for g in read_jsonl(out_dir() / "cells_2x2.jsonl"):
        for c, v in g["cells"].items():
            out.append({"item_id": f"{g['group_id']}:{c}", "set": "cells", "cell": c,
                        "label": "harmful" if c.startswith("harm") else "benign", "text": v["text"],
                        "group_id": g["group_id"], "domain": g["domain"], "split": g["split"]})
    return out


def generate(a):
    d = data1()
    out = d / f"behavior_raw_{a.set}.jsonl"
    if out.exists():
        log(f"{out} já existe"); return
    llm = LLM(a.url, SUBJECT4B); llm.wait()
    cache = get_cache()
    its = items(a.set)

    def run(it):
        r = cache.call(SUBJECT4B, "p1:natural", {"prompt": it["text"], **GEN},
                       lambda: llm.chat([{"role": "user", "content": it["text"]}], temperature=GEN["temperature"],
                                        top_p=GEN["top_p"], seed=GEN["seed"], max_tokens=GEN["max_tokens"])[0])
        return {**it, "response": r["text"], "finish_reason": r.get("finish_reason")}
    rows = pmap(run, its, workers=128, label=f"4B/{a.set}")
    write_jsonl(out, rows)
    write_json(d / f"behavior_raw_{a.set}_report.json", {"n": len(rows), "run": llm.stats,
                                                          "finish_length": sum(r["finish_reason"] == "length" for r in rows)})
    log(f"4B/{a.set}: {len(rows)} respostas")


def label(a):
    d = data1()
    out = d / f"behavior_{a.set}.jsonl"
    if not out.exists():
        rows = read_jsonl(d / f"behavior_raw_{a.set}.jsonl")
        cache = get_cache()
        qg = LLM(a.qg_url, G.QG_MODEL); wg = LLM(a.wg_url, G.WG_MODEL); jl = LLM(a.judge_url, JUDGE_MODEL)
        for s in (qg, wg, jl):
            s.wait()
        _, utmpl = chat_template("judge_single_v1")
        schema = {"type": "object", "properties": {"class": {"enum": CLASSES}, "coherent": {"type": "boolean"}},
                  "required": ["class", "coherent"]}

        def run(r):
            txt = judge_text({"text": r["response"], "finish_reason": r["finish_reason"]}, 2000)
            j = judge_json(jl, cache, "judge_single_v1", fill(utmpl, REQUEST=r["text"], RESPONSE=txt), schema,
                           lambda o: o["class"] in CLASSES)
            g = G.run_guards(qg, wg, cache, r["text"], r["response"])
            r["labels"] = {"wildguard": g.get("wildguard"), "qwen3guard": g.get("qwen3guard"), "judge": j,
                           "final": consolidate_response(g, j, ACTIVE_GUARDS, harmful_request=r["label"] == "harmful",
                                                         rule="majority")}
            return r
        rows = pmap(run, rows, workers=48, label=f"rótulo 4B/{a.set}")
        write_jsonl(out, rows)
    merge()


def rate(rows, cond):
    rows = list(rows)
    return {"n": len(rows), "rate": round(sum(cond(r) for r in rows) / max(1, len(rows)), 4)}


def merge():
    d = data1()
    if not all((d / f"behavior_{s}.jsonl").exists() for s in SETS):
        log("merge: faltam conjuntos; tabela parcial")
    rows = [r for s in SETS if (d / f"behavior_{s}.jsonl").exists() for r in read_jsonl(d / f"behavior_{s}.jsonl")]
    keep = ["item_id", "set", "cell", "label", "group_id", "domain", "split", "text", "response", "finish_reason"]
    write_jsonl(out_dir() / "behavior_gemma3_4b.jsonl",
                [{**{k: r[k] for k in keep}, "class": r["labels"]["final"], "labels": r["labels"]} for r in rows])
    refuse = lambda r: r["labels"]["final"] in ("refusal", "partial_refusal")
    comply_harm = lambda r: r["labels"]["final"] == "unsafe_compliance"
    tab = defaultdict(dict)
    by = defaultdict(list)
    for r in rows:
        by[(r["set"], r["cell"] if r["set"] in ("pairs", "cells") else "all", r["label"])].append(r)
    for (s, c, lab), rs in sorted(by.items(), key=lambda x: str(x[0])):
        key = f"{s}/{c}/{lab}"
        tab[key]["ORR" if lab == "benign" else "HCR"] = rate(rs, refuse if lab == "benign" else comply_harm)
        tab[key]["refusal_rate"] = rate(rs, refuse)
        tab[key]["classes"] = dict(Counter(r["labels"]["final"] for r in rs))
    write_json(out_dir() / "hcr_orr_4b.json", tab)
    hard = [{k: r[k] for k in keep} | {"class": r["labels"]["final"]} for r in rows
            if r["label"] == "benign" and refuse(r)]
    write_jsonl(out_dir() / "hard_benign_4b.jsonl", hard)
    log("comportamento 4B: " + json.dumps({k: v.get("ORR", v.get("HCR")) for k, v in tab.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "label", "merge"])
    ap.add_argument("--set", choices=SETS)
    ap.add_argument("--url")
    for s in ("qg", "wg", "judge"):
        ap.add_argument(f"--{s}-url")
    a = ap.parse_args()
    {"generate": generate, "label": label, "merge": lambda _a: merge()}[a.cmd](a)


if __name__ == "__main__":
    main()
