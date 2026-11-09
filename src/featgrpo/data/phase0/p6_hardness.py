"""P6 — dificuldade de Q⁰: fração dos 4 modelos de referência que recusam (total ou parcialmente) (§P6).

    python -m featgrpo.data.phase0.p6_hardness generate --model M --url U   # todos os Q⁰ de groups_p3
    python -m featgrpo.data.phase0.p6_hardness label --qg-url U --wg-url U --judge-url U   # só grupos válidos

A geração cobre todos os grupos (não só os válidos) para rodar em paralelo com o P4; o rótulo e a
dificuldade só são calculados para os grupos válidos do P5, como no plano.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from .common import LLM, data_dir, get_cache, load_config, log, pmap, read_jsonl, write_json, write_jsonl

REFUSING = {"refusal", "partial_refusal"}


def hardness(ref_classes: dict) -> float:
    if not ref_classes:
        return 0.0
    return sum(c in REFUSING for c in ref_classes.values()) / len(ref_classes)


def short(model: str) -> str:
    return model.split("/")[-1]


def generate(a, cfg):
    from .p3_responses import generate as gen_one
    d = data_dir()
    cache = get_cache()
    llm = LLM(a.url, a.model)
    llm.wait()
    src = d / ("groups_p5.jsonl" if a.valid_only else "groups_p3.jsonl")
    groups = [g for g in read_jsonl(src) if not a.valid_only or g.get("valid")]
    outs = pmap(lambda g: gen_one(llm, cache, cfg, a.model, g["q_border"]["text"], None, 0), groups,
                workers=cfg["generation"].get("concurrency", 64), label=f"P6 {short(a.model)}")
    (d / "p6").mkdir(exist_ok=True)
    write_json(d / "p6" / f"{short(a.model)}.json", {g["group_id"]: o for g, o in zip(groups, outs)})
    log(f"P6 generate ok {a.model} {len(groups)} {llm.stats}")


def label(a, cfg):
    from .p4_label import active_guards, label_group, servers
    d = data_dir()
    qg, wg, jl, judge, cache = servers(a, cfg)
    active = active_guards()
    groups = read_jsonl(d / "groups_p5.jsonl")
    refs = {short(m): json.loads((d / "p6" / f"{short(m)}.json").read_text()) for m in cfg["models"]["reference"]}

    def run(g):
        if not g["valid"]:
            return
        rs = [{"id": f"{g['group_id']}_ref_{m}", "q_role": "border", "role": "natural", "generator": m,
               "text": refs[m][g["group_id"]]["text"], "finish_reason": refs[m][g["group_id"]]["finish_reason"]}
              for m in refs]
        gg = dict(g)
        gg["responses"] = rs
        label_group(gg, qg, wg, judge, cache, cfg, active, only={r["id"] for r in rs})
        g["q_border"]["ref_classes"] = {r["generator"]: r["labels"]["final"] for r in rs}
        g["q_border"]["ref_responses"] = rs
        g["q_border"]["hardness"] = hardness(g["q_border"]["ref_classes"])

    pmap(run, groups, workers=cfg["models"]["judge"].get("concurrency", 32), label="P6 rótulos")
    write_jsonl(d / "groups_p6.jsonl", groups)
    valid = [g for g in groups if g["valid"]]
    rep = {"valid_groups": len(valid),
           "hardness_distribution": dict(sorted(Counter(str(g["q_border"]["hardness"]) for g in valid).items())),
           "hardness_by_domain": {dom: dict(sorted(Counter(str(g["q_border"]["hardness"]) for g in valid
                                                         if g["domain"] == dom).items()))
                                  for dom in cfg["domain_priority"]},
           "ref_class_counts": {m: dict(Counter(g["q_border"]["ref_classes"][m] for g in valid)) for m in refs},
           "unresolved": sum(any(c in ("disputed", "incoherent") for c in g["q_border"]["ref_classes"].values())
                             for g in valid)}
    write_json(d / "groups_p6_report.json", rep)
    log(f"P6 ok {json.dumps(rep)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "label"])
    ap.add_argument("--model")
    ap.add_argument("--url")
    ap.add_argument("--valid-only", action="store_true")
    for n in ("qg", "wg", "judge"):
        ap.add_argument(f"--{n}-url")
    a = ap.parse_args()
    cfg = load_config()
    generate(a, cfg) if a.cmd == "generate" else label(a, cfg)


if __name__ == "__main__":
    main()
