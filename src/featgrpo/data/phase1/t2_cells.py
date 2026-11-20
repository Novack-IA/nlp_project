"""A-P1/A-P3: 2×2 por frase-molde. Gera moldes (gemma-4-31B), monta as 4 células, valida
estruturalmente e rotula (WildGuard + Qwen3Guard + juiz 70B, maioria 2/3) em rodadas: o candidato k
só é tentado nos grupos em que os anteriores falharam.

    python -m featgrpo.data.phase1.t2_cells --gen-url U --qg-url U --wg-url U --judge-url U --tag dry --n 50
    python -m featgrpo.data.phase1.t2_cells ... --tag full

Células: harm_trig, harm_notrig, ben_trig, ben_notrig (h = intenção, t = forma alarmante).
Saídas (PHASE1_DATA): cells_<tag>_all.jsonl (todos, com motivos), cells_<tag>_report.json;
tag=full também escreve datasets/phase1/cells_2x2.jsonl (só os aprovados).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from ..phase0 import guards as G
from ..phase0.common import LLM, get_cache, no_thinking, pmap, read_jsonl, write_json, write_jsonl
from ..phase0.p4_label import consolidate_intent
from .common1 import (ACTIVE_GUARDS, GEN_MODEL, JUDGE_MODEL, PAIRS, chat_template, contains_phrase, data1, fill,
                      hfrac, judge_json, log, n_words, out_dir, stems)

CELLS = ["harm_trig", "harm_notrig", "ben_trig", "ben_notrig"]
K = 3                 # candidatos de molde por grupo
LEN_RATIO = 1.20      # máx/mín de palavras entre as 4 células (proposta: ≤ 20%)
LEN_SLACK = 3         # ...ou diferença absoluta ≤ 3 palavras (pedidos curtos; desvio registrado)


# ---------------------------------------------------------------- geração

def gen_candidates(llm: LLM, cache, p: dict, ttype: str | None) -> list[dict]:
    system, utmpl = chat_template("cell_template_v1")
    user = fill(utmpl, QH=p["q_harm"], QB=p["q_border"], TRIGGER=p["trigger"], TTYPE=ttype or "surface")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    out = []
    for k in range(K):
        r = cache.call(GEN_MODEL, "p1:cell_template_v1", {"user": user, "seed": k, "t": 0.7},
                       lambda: llm.chat(msgs, temperature=0.7, top_p=0.95, seed=k, max_tokens=400,
                                        extra=no_thinking(GEN_MODEL))[0])
        try:
            from ..phase0.common import extract_json
            o = extract_json(r["text"])
            out.append({k2: str(o.get(k2, "")).strip() for k2 in
                        ("frame", "form_alarm", "form_neutral", "intent_harmful", "intent_benign")})
        except (ValueError, AttributeError):
            out.append(None)
    return out


# ---------------------------------------------------------------- montagem e validação estrutural

def assemble(c: dict) -> dict:
    """Preenche o molde; devolve textos e spans (char) das lacunas em cada célula."""
    fr = c["frame"]
    i_f, i_i = fr.index("{FORM}"), fr.index("{INTENT}")
    cells = {}
    for name, form, intent in (("harm_trig", c["form_alarm"], c["intent_harmful"]),
                               ("harm_notrig", c["form_neutral"], c["intent_harmful"]),
                               ("ben_trig", c["form_alarm"], c["intent_benign"]),
                               ("ben_notrig", c["form_neutral"], c["intent_benign"])):
        parts = sorted([(i_f, "{FORM}", form, "form"), (i_i, "{INTENT}", intent, "intent")])
        text, spans, last = "", {}, 0
        for pos, ph, val, key in parts:
            text += fr[last:pos]
            spans[key] = [len(text), len(text) + len(val)]
            text += val
            last = pos + len(ph)
        text += fr[last:]
        # capitaliza início se o molde começa por lacuna
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        cells[name] = {"text": text, "slots": spans}
    return cells


def structural_check(c: dict | None, p: dict, ttype: str | None) -> tuple[bool, str, dict | None]:
    if not c:
        return False, "json_invalido", None
    fr = c["frame"]
    if fr.count("{FORM}") != 1 or fr.count("{INTENT}") != 1:
        return False, "lacunas", None
    if not all(c[k] for k in ("form_alarm", "form_neutral", "intent_harmful", "intent_benign")):
        return False, "filler_vazio", None
    fa, fn = c["form_alarm"], c["form_neutral"]
    if stems(fa) & stems(fn) or fa.lower() == fn.lower():
        return False, "forma_neutra_compartilha_radical", None
    rest = fr.replace("{FORM}", " ").replace("{INTENT}", " ")
    for piece in (rest, c["intent_harmful"], c["intent_benign"]):
        if contains_phrase(piece, fa):
            return False, "forma_alarmante_fora_da_lacuna", None
        if fn.lower() in piece.lower():
            return False, "forma_neutra_fora_da_lacuna", None
    if ttype in ("surface", "action") and not contains_phrase(fa, p["trigger"]) and not contains_phrase(p["trigger"], fa):
        return False, "forma_alarmante_nao_e_o_gatilho", None
    cells = assemble(c)
    for name in ("harm_notrig", "ben_notrig"):
        if contains_phrase(cells[name]["text"], fa):
            return False, "gatilho_presente_sem_gatilho", None
    lens = [n_words(cells[n]["text"]) for n in CELLS]
    if min(lens) == 0 or (max(lens) / min(lens) > LEN_RATIO and max(lens) - min(lens) > LEN_SLACK):
        return False, "tamanho", None
    if max(lens) > 60:
        return False, "longo_demais", None
    return True, "", cells


# ---------------------------------------------------------------- rotulagem

def label_cells(qg, wg, jl, cache, p: dict, cells: dict) -> dict:
    rnd = random.Random(int(hfrac(p["group_id"]) * 1e9))
    order = CELLS[:]
    rnd.shuffle(order)
    rid = {name: f"R{i + 1}" for i, name in enumerate(order)}
    reqs = [{"id": rid[n], "text": cells[n]["text"]} for n in order] + \
           [{"id": "OA", "text": p["q_harm"]}, {"id": "OB", "text": p["q_border"]}]
    pairs = [("P1", rid["harm_trig"], rid["harm_notrig"]), ("P2", rid["ben_trig"], rid["ben_notrig"]),
             ("P3", rid["harm_trig"], "OA"), ("P4", rid["ben_trig"], "OB")]
    _, utmpl = chat_template("cell_judge_v1")
    user = fill(utmpl, REQUESTS=json.dumps(reqs, ensure_ascii=False, indent=1),
                PAIRS=json.dumps([{"pair": a, "requests": [b, c]} for a, b, c in pairs], ensure_ascii=False, indent=1))
    ids = [r["id"] for r in reqs]
    pids = [x[0] for x in pairs]
    schema = {"type": "object", "properties": {
        "requests": {"type": "array", "minItems": len(ids), "maxItems": len(ids), "items": {"type": "object",
                     "properties": {"id": {"enum": ids}, "intent": {"enum": ["harmful", "benign"]},
                                    "natural": {"type": "boolean"}}, "required": ["id", "intent", "natural"]}},
        "pairs": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "object",
                  "properties": {"pair": {"enum": pids}, "same_request": {"type": "boolean"}},
                  "required": ["pair", "same_request"]}}}, "required": ["requests", "pairs"]}
    ok = lambda o: {x["id"] for x in o["requests"]} == set(ids) and {x["pair"] for x in o["pairs"]} == set(pids)
    j = judge_json(jl, cache, "cell_judge_v1", user, schema, ok)
    jr = {x["id"]: x for x in j["requests"]} if j else {}
    jp = {x["pair"]: x["same_request"] for x in j["pairs"]} if j else {}
    lab = {"judge_error": j is None, "same": {"harm_row": jp.get("P1"), "ben_row": jp.get("P2"),
                                              "harm_vs_orig": jp.get("P3"), "ben_vs_orig": jp.get("P4")},
           "cells": {}}
    for n in CELLS:
        g = G.run_guards(qg, wg, cache, cells[n]["text"], None)
        ji = jr.get(rid[n], {})
        lab["cells"][n] = {"wildguard": g.get("wildguard"), "qwen3guard": g.get("qwen3guard"),
                           "judge": ji.get("intent"), "natural": ji.get("natural"),
                           "final": consolidate_intent(g, ji.get("intent"), ACTIVE_GUARDS, "majority")}
    return lab


def verdict(lab: dict) -> tuple[bool, str]:
    if lab["judge_error"]:
        return False, "erro_juiz"
    for n in CELLS:
        want = "harmful" if n.startswith("harm") else "benign"
        if lab["cells"][n]["final"] != want:
            return False, f"intencao:{n}={lab['cells'][n]['final']}"
        if lab["cells"][n]["natural"] is False:
            return False, f"nao_natural:{n}"
    for k, v in lab["same"].items():
        if v is not True:
            return False, f"mudou_pedido:{k}"
    return True, ""


# ---------------------------------------------------------------- principal

def select(pairs: list[dict], tag: str, n: int | None) -> list[dict]:
    if tag != "dry":
        return pairs
    by = {}
    for p in sorted(pairs, key=lambda p: hfrac("dry:" + p["group_id"])):
        if p["split"] != "test":
            by.setdefault(p["domain"], []).append(p)
    per = -(-(n or 50) // len(by))
    return [p for d in sorted(by) for p in by[d][:per]][: n or 50]


def main():
    ap = argparse.ArgumentParser()
    for s in ("gen", "qg", "wg", "judge"):
        ap.add_argument(f"--{s}-url", required=True)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    d = data1()
    allp = d / f"cells_{a.tag}_all.jsonl"
    if allp.exists():
        log(f"t2: {allp} já existe"); return
    cache = get_cache()
    gen = LLM(a.gen_url, GEN_MODEL); qg = LLM(a.qg_url, G.QG_MODEL); wg = LLM(a.wg_url, G.WG_MODEL)
    jl = LLM(a.judge_url, JUDGE_MODEL)
    for s in (gen, qg, wg, jl):
        s.wait()
    tt_path = out_dir() / "trigger_types.jsonl"
    ttypes = {r["trigger"]: r["type"] for r in read_jsonl(tt_path)} if tt_path.exists() else {}
    pairs = select(read_jsonl(PAIRS), a.tag, a.n)
    log(f"t2[{a.tag}]: {len(pairs)} grupos; tipologia {'ok' if ttypes else 'ausente'}")

    cands = pmap(lambda p: gen_candidates(gen, cache, p, ttypes.get(p["trigger"].lower())), pairs, workers=96,
                 label=f"t2[{a.tag}]/molde")
    state = {p["group_id"]: {"pair": p, "cands": c, "tries": [], "ok": False} for p, c in zip(pairs, cands)}
    for k in range(K):
        todo = []
        for gid, s in state.items():
            if s["ok"]:
                continue
            p = s["pair"]
            good, why, cells = structural_check(s["cands"][k], p, ttypes.get(p["trigger"].lower()))
            if not good:
                s["tries"].append({"k": k, "stage": "estrutura", "reason": why, "cand": s["cands"][k]})
            else:
                todo.append((gid, cells))
        log(f"t2[{a.tag}] rodada {k}: {len(todo)} candidatos passam na estrutura")
        labs = pmap(lambda x: label_cells(qg, wg, jl, cache, state[x[0]]["pair"], x[1]), todo, workers=32,
                    label=f"t2[{a.tag}]/rótulo{k}")
        for (gid, cells), lab in zip(todo, labs):
            s = state[gid]
            good, why = verdict(lab)
            s["tries"].append({"k": k, "stage": "rotulo", "reason": why, "cand": s["cands"][k], "labels": lab})
            if good:
                s["ok"] = True
                s["final"] = {"k": k, "cand": s["cands"][k], "cells": cells, "labels": lab}

    rows = []
    for gid, s in state.items():
        p = s["pair"]
        r = {"group_id": gid, "domain": p["domain"], "subdomain": p.get("subdomain"), "split": p["split"],
             "trigger": p["trigger"], "trigger_type": ttypes.get(p["trigger"].lower()),
             "orig": {"q_harm": p["q_harm"], "q_border": p["q_border"], "q_border_method": p["q_border_method"]},
             "ok": s["ok"], "tries": s["tries"]}
        if s["ok"]:
            f = s["final"]
            r.update({"candidate": f["k"], "frame": f["cand"]["frame"], "form_alarm": f["cand"]["form_alarm"],
                      "form_neutral": f["cand"]["form_neutral"], "intent_harmful": f["cand"]["intent_harmful"],
                      "intent_benign": f["cand"]["intent_benign"],
                      "cells": {n: {**f["cells"][n], "labels": f["labels"]["cells"][n]} for n in CELLS},
                      "same": f["labels"]["same"]})
        rows.append(r)
    write_jsonl(allp, rows)

    ok = [r for r in rows if r["ok"]]
    first_fail = Counter()
    for r in rows:
        if not r["ok"]:
            first_fail[(r["tries"][0]["reason"] if r["tries"] else "?").split(":")[0]] += 1
    all_fail = Counter(t["reason"].split("=")[0] for r in rows for t in r["tries"] if t["reason"])
    rep = {"tag": a.tag, "groups": len(rows), "ok": len(ok), "yield": len(ok) / max(1, len(rows)),
           "ok_by_candidate": dict(Counter(r["candidate"] for r in ok)),
           "by_domain": {dm: {"n": sum(r["domain"] == dm for r in rows), "ok": sum(r["domain"] == dm for r in ok)}
                         for dm in sorted({r["domain"] for r in rows})},
           "by_trigger_type": {str(t): {"n": sum(r["trigger_type"] == t for r in rows),
                                        "ok": sum(r["trigger_type"] == t for r in ok)}
                               for t in sorted({str(r["trigger_type"]) for r in rows})},
           "failure_reasons_all_tries": dict(all_fail.most_common()),
           "split_ok": dict(Counter(r["split"] for r in ok)),
           "run": {"gen": gen.stats, "judge": jl.stats, "qg": qg.stats, "wg": wg.stats}}
    write_json(d / f"cells_{a.tag}_report.json", rep)
    if a.tag == "full":
        out = [{k: v for k, v in r.items() if k not in ("tries", "ok")} for r in ok]
        write_jsonl(out_dir() / "cells_2x2.jsonl", out)
    log(f"t2[{a.tag}] ok: {len(ok)}/{len(rows)} ({rep['yield']:.0%}) {json.dumps(rep['by_domain'])}")


if __name__ == "__main__":
    main()
