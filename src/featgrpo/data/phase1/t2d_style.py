"""Caminho (b) do 2×2: plano B com reescrita em ESTILO COMUM, como a tarefa A1 previa.

No 2×2 entregue (`t2b_swap`), harm_trig é o Q⁺ do PKU e ben_trig é o Q⁰ do gerador: a linha nociva e a
benigna diferem também em fonte e estilo (AUROC 0,88 do classificador só de estilo). Aqui:

  1. o gerador reescreve Q⁺ e Q⁰ JUNTOS num estilo comum (prompt style_v1): uma pergunta direta, mesma
     abertura e estrutura, mesmo tamanho, o gatilho literal nos dois, sem mudar o pedido;
  2. checagem estrutural da reescrita (gatilho literal nas duas, termina em "?", uma frase, tamanho);
  3. a mesma troca só do trecho do gatilho do `t2b_swap` (swap_v1), sobre o par reescrito;
  4. os mesmos rótulos do `t2b_swap` (`label_cells`): intenção 2/3 nas 4 células, naturalidade, mesmo pedido
     dentro de cada linha e **cada linha igual ao pedido ORIGINAL** (P3/P4 comparam com o Q⁺/Q⁰ da Fase 0).

Até K reescritas por grupo; em cada uma, até K candidatos de troca. Fica o primeiro que passa em tudo.

    python -m featgrpo.data.phase1.t2d_style --gen-url U --qg-url U --wg-url U --judge-url U --tag style_smoke --n 40
    python -m featgrpo.data.phase1.t2d_style ... --tag style_full   (escreve datasets/phase1/cells_2x2_style.jsonl)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from ..phase0 import guards as G
from ..phase0.common import LLM, extract_json, get_cache, no_thinking, pmap, read_jsonl, write_json, write_jsonl
from .common1 import GEN_MODEL, JUDGE_MODEL, PAIRS, chat_template, data1, fill, hfrac, log, n_words, out_dir
from .t2_cells import CELLS, K, label_cells, verdict
from .t2b_swap import build, gen_candidates, locate

_SENT = re.compile(r"[.!?]\s+[A-Z]")


def select_style(pairs: list[dict], tag: str, n: int | None) -> list[dict]:
    """smoke: n grupos estratificados por domínio, fora do test (mesma regra das rodadas secas)."""
    if "smoke" not in tag:
        return pairs
    by: dict[str, list] = {}
    for p in sorted(pairs, key=lambda p: hfrac("style:" + p["group_id"])):
        if p["split"] != "test":
            by.setdefault(p["domain"], []).append(p)
    per = -(-(n or 40) // len(by))
    return [p for d in sorted(by) for p in by[d][:per]][: n or 40]


def style_candidate(llm: LLM, cache, p: dict, k: int) -> dict | None:
    system, utmpl = chat_template("style_v1")
    user = fill(utmpl, QH=p["q_harm"], QB=p["q_border"], TRIGGER=p["trigger"])
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    t = 0.0 if k == 0 else 0.7
    r = cache.call(GEN_MODEL, "p1:style_v1", {"user": user, "seed": k, "t": t},
                   lambda: llm.chat(msgs, temperature=t, top_p=0.95, seed=k, max_tokens=200,
                                    extra=no_thinking(GEN_MODEL))[0])
    try:
        o = extract_json(r["text"])
        a, b = str(o["A"]).strip(), str(o["B"]).strip()
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    return {"q_harm": a, "q_border": b} if a and b else None


def style_check(p: dict, s: dict | None) -> str:
    """'' se a reescrita passa; senão o motivo."""
    if s is None:
        return "json_invalido"
    trig = p["trigger"].lower()
    for key, q in (("A", s["q_harm"]), ("B", s["q_border"])):
        if q.lower().count(trig) != 1:
            return f"gatilho_literal:{key}"
        if not q.endswith("?"):
            return f"nao_pergunta:{key}"
        if _SENT.search(q):
            return f"mais_de_uma_frase:{key}"
        if not 6 <= n_words(q) <= 36:
            return f"tamanho_absoluto:{key}"
    la, lb = n_words(s["q_harm"]), n_words(s["q_border"])
    if not (max(la, lb) / min(la, lb) <= 1.2 or abs(la - lb) <= 3):
        return "tamanho_entre_linhas"
    if s["q_harm"].strip().lower() == s["q_border"].strip().lower():
        return "iguais"
    return ""


def run_group(gen, qg, wg, jl, cache, p: dict) -> dict:
    tries = []
    for ks in range(K):
        s = style_candidate(gen, cache, p, ks)
        why = style_check(p, s)
        if why:
            tries.append({"ks": ks, "stage": "estilo", "reason": why, "styled": s})
            continue
        sp = {**p, **s}                                  # par reescrito (a troca opera sobre ele)
        spans = {"A": locate(sp["q_harm"], p["trigger"]), "B": locate(sp["q_border"], p["trigger"])}
        if spans["A"] is None or spans["B"] is None:
            tries.append({"ks": ks, "stage": "localizar", "reason": "gatilho_nao_localizado", "styled": s})
            continue
        cands = gen_candidates(gen, cache, sp, sp["q_harm"][slice(*spans["A"])], sp["q_border"][slice(*spans["B"])])
        for kw, c in enumerate(cands):
            good, why, cells = build(sp, spans, c)
            if not good:
                tries.append({"ks": ks, "kw": kw, "stage": "estrutura", "reason": why, "styled": s, "cand": c})
                continue
            lab = label_cells(qg, wg, jl, cache, p, cells)   # p ORIGINAL: P3/P4 = mesmo pedido do Q⁺/Q⁰ da Fase 0
            good, why = verdict(lab)
            tries.append({"ks": ks, "kw": kw, "stage": "rotulo", "reason": why, "styled": s, "cand": c})
            if good:
                return {"ok": True, "tries": tries,
                        "final": {"ks": ks, "kw": kw, "styled": s, "cells": cells, "labels": lab}}
            break   # o rótulo reprovou a reescrita ou a troca: tenta outra reescrita
    return {"ok": False, "tries": tries}


def main():
    ap = argparse.ArgumentParser()
    for s in ("gen", "qg", "wg", "judge"):
        ap.add_argument(f"--{s}-url", required=True)
    ap.add_argument("--tag", default="style_full")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    d = data1()
    allp = d / f"cells_{a.tag}_all.jsonl"
    if allp.exists():
        log(f"t2d: {allp} já existe"); return
    cache = get_cache()
    gen = LLM(a.gen_url, GEN_MODEL); qg = LLM(a.qg_url, G.QG_MODEL); wg = LLM(a.wg_url, G.WG_MODEL)
    jl = LLM(a.judge_url, JUDGE_MODEL)
    for s in (gen, qg, wg, jl):
        s.wait()
    tt_path = out_dir() / "trigger_types.jsonl"
    ttypes = {r["trigger"]: r["type"] for r in read_jsonl(tt_path)} if tt_path.exists() else {}
    pairs = select_style(read_jsonl(PAIRS), a.tag, a.n)
    log(f"t2d[{a.tag}]: {len(pairs)} grupos")
    res = pmap(lambda p: run_group(gen, qg, wg, jl, cache, p), pairs, workers=a.workers, label=f"t2d[{a.tag}]")

    rows = []
    for p, s in zip(pairs, res):
        r = {"group_id": p["group_id"], "design": "swap_style", "domain": p["domain"], "subdomain": p.get("subdomain"),
             "split": p["split"], "trigger": p["trigger"], "trigger_type": ttypes.get(p["trigger"].lower()),
             "orig": {"q_harm": p["q_harm"], "q_border": p["q_border"], "q_border_method": p["q_border_method"]},
             "ok": s["ok"], "tries": s["tries"]}
        if s["ok"]:
            f = s["final"]
            r.update({"styled": f["styled"], "candidate": {"style": f["ks"], "swap": f["kw"]}, "frame": None,
                      "form_alarm": f["cells"]["harm_trig"]["form"], "form_neutral": f["cells"]["harm_notrig"]["form"],
                      "form_alarm_ben": f["cells"]["ben_trig"]["form"], "form_neutral_ben": f["cells"]["ben_notrig"]["form"],
                      "cells": {n: {"text": f["cells"][n]["text"], "slots": f["cells"][n]["slots"],
                                    "labels": f["labels"]["cells"][n]} for n in CELLS},
                      "same": f["labels"]["same"]})
        rows.append(r)
    write_jsonl(allp, rows)
    ok = [r for r in rows if r["ok"]]

    def key(reason: str) -> str:
        head = reason.split("=")[0]
        return head if head.startswith(("intencao", "mudou", "nao_natural")) else head.split(":")[0]
    fails = Counter(f"{t['stage']}/{key(t['reason'])}" for r in rows for t in r["tries"] if t["reason"])
    lens = [abs(n_words(r["cells"]["harm_trig"]["text"]) - n_words(r["cells"]["ben_trig"]["text"])) for r in ok]
    rep = {"tag": a.tag, "design": "swap_style", "groups": len(rows), "ok": len(ok), "yield": len(ok) / max(1, len(rows)),
           "ok_by_candidate": dict(Counter(f"s{r['candidate']['style']}w{r['candidate']['swap']}" for r in ok)),
           "by_domain": {dm: {"n": sum(r["domain"] == dm for r in rows), "ok": sum(r["domain"] == dm for r in ok)}
                         for dm in sorted({r["domain"] for r in rows})},
           "by_trigger_type": {str(t): {"n": sum(r["trigger_type"] == t for r in rows),
                                        "ok": sum(r["trigger_type"] == t for r in ok)}
                               for t in sorted({str(r["trigger_type"]) for r in rows})},
           "split_ok": dict(Counter(r["split"] for r in ok)),
           "failure_reasons_all_tries": dict(fails.most_common()),
           "len_diff_words_between_rows": dict(sorted(Counter(lens).items())),
           "run": {"gen": gen.stats, "judge": jl.stats, "qg": qg.stats, "wg": wg.stats}}
    write_json(d / f"cells_{a.tag}_report.json", rep)
    clean = [{k: v for k, v in r.items() if k not in ("tries", "ok")} for r in ok]
    write_jsonl(d / f"cells_{a.tag}.jsonl", clean)            # aprovados (o t2c_qc anota com --path)
    if a.tag == "style_full":
        write_jsonl(out_dir() / "cells_2x2_style.jsonl", clean)
    log(f"t2d[{a.tag}] ok: {len(ok)}/{len(rows)} ({rep['yield']:.0%}) {json.dumps(rep['by_domain'])}")


if __name__ == "__main__":
    main()
