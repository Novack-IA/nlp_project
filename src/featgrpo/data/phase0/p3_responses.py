"""P3 — respostas do sujeito (gemma-4-12B-it) para Q⁺ e Q⁰ (§P3).

Cada grupo fica com 8 respostas:
  r0 unsafe (PKU) · r1 safe_A (PKU) · r2 safe_helpful(Q⁺) · r3 refusal(Q⁺) · r4 natural(Q⁺)
  r5 safe_helpful(Q⁰) · r6 refusal(Q⁰) · r7 natural(Q⁰)

    python -m featgrpo.data.phase0.p3_responses --url URL
"""
from __future__ import annotations

import argparse

from .common import (LLM, data_dir, get_cache, load_config, load_prompt, log, no_thinking, pmap, read_jsonl,
                     strip_thinking, write_json, write_jsonl)

SUBJECT_ITEMS = [("r2", "harm", "safe_helpful", "safe_helpful_v1"), ("r3", "harm", "refusal", "refusal_v1"),
                 ("r4", "harm", "natural", None), ("r5", "border", "safe_helpful", "safe_helpful_v1"),
                 ("r6", "border", "refusal", "refusal_v1"), ("r7", "border", "natural", None)]


def generate(llm: LLM, cache, cfg: dict, model: str, prompt: str, system_id: str | None, seed: int) -> dict:
    gc = cfg["generation"]
    msgs = ([{"role": "system", "content": load_prompt(cfg, system_id).strip()}] if system_id else []) + \
           [{"role": "user", "content": prompt}]
    r = cache.call(model, f"respond:{system_id}", {"prompt": prompt, "seed": seed, "max": gc["max_new_tokens"],
                                                   "t": gc["temperature"], "p": gc["top_p"]},
                   lambda: llm.chat(msgs, temperature=gc["temperature"], top_p=gc["top_p"], seed=seed,
                                    max_tokens=gc["max_new_tokens"], extra=no_thinking(model))[0])
    text, had = strip_thinking(r["text"])
    return {"text": text, "finish_reason": r.get("finish_reason"), "thinking_stripped": had}


def build_responses(g: dict, gen: dict) -> list[dict]:
    gid = g["group_id"]
    rs = [{"id": f"{gid}_r0", "q_role": "harm", "role": "unsafe", "generator": g["generator"],
           "system_prompt_id": None, "seed": None, "text": g["r_unsafe"], "finish_reason": None},
          {"id": f"{gid}_r1", "q_role": "harm", "role": "safe_A", "generator": g["generator"],
           "system_prompt_id": None, "seed": None, "text": g["r_safe_A"], "finish_reason": None}]
    for rid, qrole, role, sp in SUBJECT_ITEMS:
        o = gen[rid]
        rs.append({"id": f"{gid}_{rid}", "q_role": qrole, "role": role, "generator": "subject",
                   "system_prompt_id": sp, "seed": 0, "text": o["text"], "finish_reason": o["finish_reason"],
                   "thinking_stripped": o["thinking_stripped"]})
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    a = ap.parse_args()
    cfg = load_config()
    d = data_dir()
    cache = get_cache()
    model = cfg["models"]["subject"]
    llm = LLM(a.url, model)
    llm.wait()
    groups = read_jsonl(d / "groups_p2.jsonl")
    jobs = [(g, rid, qrole, sp) for g in groups for rid, qrole, _r, sp in SUBJECT_ITEMS]

    def run(j):
        g, rid, qrole, sp = j
        q = g["q_harm"] if qrole == "harm" else g["q_border"]["text"]
        return generate(llm, cache, cfg, model, q, sp, 0)

    outs = pmap(run, jobs, workers=cfg["generation"].get("concurrency", 64), label="P3")
    by = {}
    for (g, rid, _q, _s), o in zip(jobs, outs):
        by.setdefault(g["group_id"], {})[rid] = o
    thinking = 0
    for g in groups:
        g["responses"] = build_responses(g, by[g["group_id"]])
        thinking += sum(bool(r.get("thinking_stripped")) for r in g["responses"])
    write_jsonl(d / "groups_p3.jsonl", groups)
    rep = {"groups": len(groups), "subject_responses": len(jobs), "thinking_blocks_stripped": thinking,
           "finish_length": sum(r["finish_reason"] == "length" for g in groups for r in g["responses"]),
           "run": {f"vllm:{model}:{k}": v for k, v in llm.stats.items()}}
    write_json(d / "p3_report.json", rep)
    log(f"P3 ok {rep}")


if __name__ == "__main__":
    main()
