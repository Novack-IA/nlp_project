"""A-P4 (controles) e A3 (conjuntos externos).

    python -m featgrpo.data.phase1.t3_controls_external controls
    python -m featgrpo.data.phase1.t3_controls_external external --embed-url URL

controls: para cada grupo aprovado do 2×2, as peças da lacuna de forma (form_alarm / form_neutral)
          em 3 frases-molde neutras, que não são pedidos → datasets/phase1/controls_trig_only.jsonl
external: XSTest (450), OR-Bench-Hard-1K (benignos), FalseReject-test (benignos) no formato comum,
          com o cosseno máximo contra os 4.856 pedidos dos pares → datasets/phase1/external/*.jsonl
"""
from __future__ import annotations

import argparse
import math
from collections import Counter

from ..phase0.common import LLM, raw_dir, read_jsonl, write_json, write_jsonl
from .common1 import PAIRS, data1, log, out_dir

FRAMES = [
    'The phrase "{X}" appears in the title of the third chapter.',
    'In the glossary, "{X}" is listed on page twelve.',
    'She wrote "{X}" on the whiteboard before the meeting started.',
]


def controls():
    cells = read_jsonl(out_dir() / "cells_2x2.jsonl")
    rows = []
    for g in cells:
        for i, fr in enumerate(FRAMES):
            pos = fr.index("{X}")
            for kind, x in (("trig", g["form_alarm"]), ("neutral", g["form_neutral"])):
                rows.append({"item_id": f"{g['group_id']}:ctl{i}:{kind}", "group_id": g["group_id"],
                             "template_id": i, "kind": kind, "piece": x, "text": fr.replace("{X}", x),
                             "span": [pos, pos + len(x)], "split": g["split"], "domain": g["domain"]})
    write_jsonl(out_dir() / "controls_trig_only.jsonl", rows)
    log(f"controles: {len(rows)} frases ({len(cells)} grupos × {len(FRAMES)} moldes × 2)")


def _norm(v):
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / s for x in v]


def external(embed_url: str):
    raw = raw_dir()
    sets = {
        "xstest": [{"id": f"xstest:{i}", "text": r["prompt"], "label": "benign" if r["label"] == "safe" else "harmful",
                    "type": r.get("type"), "source": "walledai/XSTest"} for i, r in enumerate(read_jsonl(raw / "xstest.jsonl"))],
        "orbench_hard": [{"id": f"orbh:{i}", "text": r["prompt"], "label": "benign", "type": r.get("category"),
                          "source": "bench-llm/or-bench-hard-1k"} for i, r in enumerate(read_jsonl(raw / "orbench_hard1k.jsonl"))],
        "falsereject": [{"id": f"frj:{i}", "text": r["prompt"], "label": "benign", "type": r.get("category_text"),
                         "source": "AmazonScience/FalseReject:test"} for i, r in enumerate(read_jsonl(raw / "falsereject_test.jsonl"))],
    }
    pairs = read_jsonl(PAIRS)
    ptexts = [p["q_harm"] for p in pairs] + [p["q_border"] for p in pairs]
    emb = LLM(embed_url, "Qwen/Qwen3-Embedding-0.6B"); emb.wait()
    import numpy as np

    def E(texts):
        out = []
        for i in range(0, len(texts), 256):
            out += emb.embed(texts[i:i + 256])
        m = np.array(out, dtype=np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)
    P = E(ptexts)
    rep = {}
    od = out_dir() / "external"; od.mkdir(exist_ok=True)
    for name, rows in sets.items():
        X = E([r["text"] for r in rows])
        mx = (X @ P.T).max(axis=1)
        for r, c in zip(rows, mx):
            r["max_cos_to_pairs"] = round(float(c), 4)
        write_jsonl(od / f"{name}.jsonl", rows)
        rep[name] = {"n": len(rows), "labels": dict(Counter(r["label"] for r in rows)),
                     "cos_gt_0.9": int((mx > 0.9).sum()), "cos_gt_0.8": int((mx > 0.8).sum())}
    write_json(data1() / "external_report.json", rep)
    log(f"externos: {rep}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["controls", "external"])
    ap.add_argument("--embed-url")
    a = ap.parse_args()
    controls() if a.cmd == "controls" else external(a.embed_url)


if __name__ == "__main__":
    main()
