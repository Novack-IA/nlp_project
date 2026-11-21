"""Controle de qualidade do 2×2 (plano B), decidido antes de qualquer ativação (ver results/DESVIOS.md).

A revisão de uma amostra mostrou dois problemas que a validação não pegava:
  - versões trocadas agramaticais (ex.: "dispose of discreetly of it");
  - termo "neutro" que não é menos alarmante (ex.: "sell drugs online" -> "illicitly distribute narcotics").
O juiz avalia a gramática das 2 versões trocadas e o nível de alarme (1–5, fora de contexto) das 4 expressões.

  qc.grammatical      = as duas versões sem gatilho são gramaticais
  qc.alarm_drop_harm  = alarme(gatilho em Q⁺) − alarme(neutro em Q⁺)   (idem _ben)
  qc_pass             = grammatical e alarm_drop ≥ 1 nas duas linhas

Não descarta nada: anota cada grupo de datasets/phase1/cells_2x2.jsonl (escrita atômica).

    python -m featgrpo.data.phase1.t2c_qc --judge-url URL
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from ..phase0.common import LLM, get_cache, pmap, read_jsonl, write_json, write_jsonl
from .common1 import JUDGE_MODEL, chat_template, data1, fill, judge_json, log, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-url", required=True)
    a = ap.parse_args()
    path = out_dir() / "cells_2x2.jsonl"
    rows = read_jsonl(path)
    jl = LLM(a.judge_url, JUDGE_MODEL); jl.wait()
    cache = get_cache()
    _, utmpl = chat_template("qc_v1")
    schema = {"type": "object", "properties": {
        "sentences": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "object", "properties": {
            "id": {"enum": ["S1", "S2"]}, "grammatical": {"type": "boolean"}}, "required": ["id", "grammatical"]}},
        "wordings": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "object", "properties": {
            "id": {"enum": ["W1", "W2", "W3", "W4"]}, "alarm": {"type": "integer", "minimum": 1, "maximum": 5}},
            "required": ["id", "alarm"]}}}, "required": ["sentences", "wordings"]}

    def run(r):
        sents = [{"id": "S1", "text": r["cells"]["harm_notrig"]["text"]}, {"id": "S2", "text": r["cells"]["ben_notrig"]["text"]}]
        words = [{"id": "W1", "text": r["form_alarm"]}, {"id": "W2", "text": r["form_neutral"]},
                 {"id": "W3", "text": r["form_alarm_ben"]}, {"id": "W4", "text": r["form_neutral_ben"]}]
        user = fill(utmpl, SENTENCES=json.dumps(sents, ensure_ascii=False, indent=1),
                    WORDINGS=json.dumps(words, ensure_ascii=False, indent=1))
        o = judge_json(jl, cache, "qc_v1", user, schema,
                       lambda o: {x["id"] for x in o["sentences"]} == {"S1", "S2"} and len(o["wordings"]) == 4)
        if o is None:
            return {"error": True}
        g = {x["id"]: x["grammatical"] for x in o["sentences"]}
        w = {x["id"]: int(x["alarm"]) for x in o["wordings"]}
        return {"grammatical_harm_notrig": g["S1"], "grammatical_ben_notrig": g["S2"],
                "alarm": {"harm_trig": w["W1"], "harm_notrig": w["W2"], "ben_trig": w["W3"], "ben_notrig": w["W4"]},
                "alarm_drop_harm": w["W1"] - w["W2"], "alarm_drop_ben": w["W3"] - w["W4"]}

    qcs = pmap(run, rows, workers=48, label="qc 2×2")
    for r, q in zip(rows, qcs):
        r["qc"] = q
        r["qc_pass"] = (not q.get("error") and q["grammatical_harm_notrig"] and q["grammatical_ben_notrig"]
                        and q["alarm_drop_harm"] >= 1 and q["alarm_drop_ben"] >= 1)
    write_jsonl(path, rows)
    ok = [r for r in rows if r["qc_pass"]]
    good = [r for r in rows if not r["qc"].get("error")]
    rep = {"groups": len(rows), "qc_pass": len(ok), "errors": len(rows) - len(good),
           "grammatical_both": sum(r["qc"]["grammatical_harm_notrig"] and r["qc"]["grammatical_ben_notrig"] for r in good),
           "drop_harm_dist": dict(sorted(Counter(r["qc"]["alarm_drop_harm"] for r in good).items())),
           "drop_ben_dist": dict(sorted(Counter(r["qc"]["alarm_drop_ben"] for r in good).items())),
           "qc_pass_by_domain": dict(Counter(r["domain"] for r in ok)),
           "qc_pass_by_split": dict(Counter(r["split"] for r in ok)),
           "qc_pass_by_trigger_type": dict(Counter(str(r["trigger_type"]) for r in ok)),
           "run": jl.stats}
    write_json(data1() / "cells_qc_report.json", rep)
    log(f"qc 2×2: {json.dumps(rep)}")


if __name__ == "__main__":
    main()
