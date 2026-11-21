"""Plano B do 2×2 (proposta do Davi, ativado porque a frase-molde rendeu < 40% na rodada seca):
mantém os pares originais e troca SÓ o trecho do gatilho por um quase-sinônimo neutro.

  harm_trig = Q⁺ original        harm_notrig = Q⁺ com o gatilho trocado
  ben_trig  = Q⁰ original        ben_notrig  = Q⁰ com o gatilho trocado

Entre com/sem gatilho muda uma única expressão (a manipulação lexical). A validação é a mesma do
molde: intenção por maioria 2/3 nas 4 células, naturalidade e "mesmo pedido" dentro de cada linha.

    python -m featgrpo.data.phase1.t2b_swap --gen-url U --qg-url U --wg-url U --judge-url U --tag swap_dry --n 50
    python -m featgrpo.data.phase1.t2b_swap ... --tag full     (escreve datasets/phase1/cells_2x2.jsonl)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from ..phase0 import guards as G
from ..phase0.common import LLM, extract_json, get_cache, no_thinking, pmap, read_jsonl, write_json, write_jsonl
from .common1 import (GEN_MODEL, JUDGE_MODEL, PAIRS, chat_template, contains_phrase, data1, fill, log, n_words,
                      out_dir, stems)
from .t2_cells import CELLS, K, label_cells, select, verdict

_TOK = re.compile(r"[A-Za-z0-9']+")


def locate(text: str, trigger: str) -> tuple[int, int] | None:
    """Span (char) do gatilho no texto: literal (sem caixa) ou a menor janela de palavras que cobre os radicais."""
    i = text.lower().find(trigger.lower())
    if i >= 0:   # estende até a borda da palavra: "term" em "terms" troca a palavra inteira (frente D, PR #8)
        a, b = i, i + len(trigger)
        while a > 0 and text[a - 1].isalnum():
            a -= 1
        while b < len(text) and text[b].isalnum():
            b += 1
        return a, b
    want = stems(trigger)
    if not want:
        return None
    toks = [(m.start(), m.end(), m.group()) for m in _TOK.finditer(text)]
    best = None
    n = len(trigger.split())
    for a in range(len(toks)):
        for b in range(a, min(len(toks), a + n + 3)):
            if want <= stems(" ".join(t[2] for t in toks[a:b + 1])):
                if best is None or (toks[b][1] - toks[a][0]) < (best[1] - best[0]):
                    best = (toks[a][0], toks[b][1])
                break
    return best


def swap(text: str, span: tuple[int, int], new: str) -> tuple[str, list[int]]:
    a, b = span
    return text[:a] + new + text[b:], [a, a + len(new)]


def gen_candidates(llm: LLM, cache, p: dict, sa: str, sb: str) -> list[dict | None]:
    system, utmpl = chat_template("swap_v1")
    user = fill(utmpl, QH=p["q_harm"], QB=p["q_border"], SPAN_A=sa, SPAN_B=sb)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    out = []
    for k in range(K):
        r = cache.call(GEN_MODEL, "p1:swap_v1", {"user": user, "seed": k, "t": 0.0 if k == 0 else 0.7},
                       lambda: llm.chat(msgs, temperature=0.0 if k == 0 else 0.7, top_p=0.95, seed=k, max_tokens=120,
                                        extra=no_thinking(GEN_MODEL))[0])
        try:
            o = extract_json(r["text"])
            out.append({"neutral_A": str(o["neutral_A"]).strip(), "neutral_B": str(o["neutral_B"]).strip()})
        except (ValueError, KeyError, TypeError, AttributeError):
            out.append(None)
    return out


def build(p: dict, spans: dict, c: dict | None) -> tuple[bool, str, dict | None]:
    if not c or not c["neutral_A"] or not c["neutral_B"]:
        return False, "json_invalido", None
    cells = {}
    for row, q, key in (("harm", p["q_harm"], "A"), ("ben", p["q_border"], "B")):
        a, b = spans[key]
        span_txt = q[a:b]
        neu = c[f"neutral_{key}"]
        if not (stems(span_txt) - stems(neu)) or neu.lower() == span_txt.lower():
            return False, "neutro_repete_o_gatilho", None
        new, sl = swap(q, (a, b), neu)
        if contains_phrase(new, p["trigger"]):
            return False, "gatilho_presente_sem_gatilho", None
        if abs(n_words(new) - n_words(q)) > 4:
            return False, "tamanho", None
        cells[f"{row}_trig"] = {"text": q, "slots": {"form": [a, b]}, "form": span_txt}
        cells[f"{row}_notrig"] = {"text": new, "slots": {"form": sl}, "form": neu}
    return True, "", cells


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
        log(f"t2b: {allp} já existe"); return
    cache = get_cache()
    gen = LLM(a.gen_url, GEN_MODEL); qg = LLM(a.qg_url, G.QG_MODEL); wg = LLM(a.wg_url, G.WG_MODEL)
    jl = LLM(a.judge_url, JUDGE_MODEL)
    for s in (gen, qg, wg, jl):
        s.wait()
    tt_path = out_dir() / "trigger_types.jsonl"
    ttypes = {r["trigger"]: r["type"] for r in read_jsonl(tt_path)} if tt_path.exists() else {}
    pairs = select(read_jsonl(PAIRS), "dry" if a.tag.endswith("dry") or a.tag.startswith("dry") else a.tag, a.n)
    log(f"t2b[{a.tag}]: {len(pairs)} grupos")

    state = {}
    todo_gen = []
    for p in pairs:
        sa, sb = locate(p["q_harm"], p["trigger"]), locate(p["q_border"], p["trigger"])
        st = {"pair": p, "tries": [], "ok": False, "spans": {"A": sa, "B": sb}}
        state[p["group_id"]] = st
        if sa is None or sb is None:
            st["tries"].append({"k": -1, "stage": "localizar", "reason": "gatilho_nao_localizado:" + ("A" if sa is None else "B")})
        else:
            todo_gen.append(p)
    cands = pmap(lambda p: gen_candidates(gen, cache, p, p["q_harm"][slice(*state[p["group_id"]]["spans"]["A"])],
                                          p["q_border"][slice(*state[p["group_id"]]["spans"]["B"])]),
                 todo_gen, workers=96, label=f"t2b[{a.tag}]/troca")
    for p, c in zip(todo_gen, cands):
        state[p["group_id"]]["cands"] = c
    for k in range(K):
        todo = []
        for gid, s in state.items():
            if s["ok"] or "cands" not in s:
                continue
            good, why, cells = build(s["pair"], s["spans"], s["cands"][k])
            if not good:
                s["tries"].append({"k": k, "stage": "estrutura", "reason": why, "cand": s["cands"][k]})
            else:
                todo.append((gid, cells))
        log(f"t2b[{a.tag}] rodada {k}: {len(todo)} candidatos passam na estrutura")
        labs = pmap(lambda x: label_cells(qg, wg, jl, cache, state[x[0]]["pair"], x[1]), todo, workers=48,
                    label=f"t2b[{a.tag}]/rótulo{k}")
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
        r = {"group_id": gid, "design": "swap", "domain": p["domain"], "subdomain": p.get("subdomain"),
             "split": p["split"], "trigger": p["trigger"], "trigger_type": ttypes.get(p["trigger"].lower()),
             "orig": {"q_harm": p["q_harm"], "q_border": p["q_border"], "q_border_method": p["q_border_method"]},
             "ok": s["ok"], "tries": s["tries"]}
        if s["ok"]:
            f = s["final"]
            r.update({"candidate": f["k"], "frame": None,
                      "form_alarm": f["cells"]["harm_trig"]["form"], "form_neutral": f["cells"]["harm_notrig"]["form"],
                      "form_alarm_ben": f["cells"]["ben_trig"]["form"], "form_neutral_ben": f["cells"]["ben_notrig"]["form"],
                      "cells": {n: {"text": f["cells"][n]["text"], "slots": f["cells"][n]["slots"],
                                    "labels": f["labels"]["cells"][n]} for n in CELLS},
                      "same": f["labels"]["same"]})
        rows.append(r)
    write_jsonl(allp, rows)
    ok = [r for r in rows if r["ok"]]
    all_fail = Counter(t["reason"].split("=")[0].split(":")[0] + (":" + t["reason"].split(":")[1].split("=")[0]
                       if t["reason"].startswith(("intencao", "mudou", "nao_natural")) else "")
                       for r in rows for t in r["tries"] if t["reason"])
    rep = {"tag": a.tag, "design": "swap", "groups": len(rows), "ok": len(ok), "yield": len(ok) / max(1, len(rows)),
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
        write_jsonl(out_dir() / "cells_2x2.jsonl", [{k: v for k, v in r.items() if k not in ("tries", "ok")} for r in ok])
    log(f"t2b[{a.tag}] ok: {len(ok)}/{len(rows)} ({rep['yield']:.0%}) {json.dumps(rep['by_domain'])}")


if __name__ == "__main__":
    main()
