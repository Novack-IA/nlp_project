"""P7 — auditoria humana (§P7).

    python -m featgrpo.data.phase0.p7_audit export    # audit/annotator_<X>.csv (sem rótulos dos juízes)
    python -m featgrpo.data.phase0.p7_audit metrics   # α por campo, κ humano–juiz, pair_valid; gera adjudication_<X>.csv
    python -m featgrpo.data.phase0.p7_audit apply     # rótulos humanos (acordo ou adjudicação) → groups_p7.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict

from .common import data_dir, load_config, log, read_jsonl, write_json, write_jsonl
from .metrics import cohen_kappa, krippendorff_alpha_nominal
from .p5_validate import REQUIRED, failures

COLS = ["group_id", "item_id", "item_type", "text", "intent", "class", "coherent", "pair_valid", "notes"]
UNRESOLVED = {"disputed"}


def gid_of(item_id: str) -> str:
    """g_000123_r0 / g_000123_q_harm → g_000123"""
    return "_".join(item_id.split("_")[:2])


def _items(g: dict, only: set[str] | None = None) -> list[dict]:
    out = []
    for key, part in (("q_harm", "A"), ("q_border", "B")):
        iid = f"{g['group_id']}_{key}"
        if only is None or iid in only:
            text = g["q_harm"] if key == "q_harm" else g["q_border"]["text"]
            out.append({"group_id": g["group_id"], "item_id": iid, "item_type": "prompt",
                        "text": f"[Request {part}] {text}" + (f"\n[Trigger] {g.get('trigger')}" if key == "q_border" else "")})
    for r in g["responses"]:
        if only is None or r["id"] in only:
            q = g["q_harm"] if r["q_role"] == "harm" else g["q_border"]["text"]
            out.append({"group_id": g["group_id"], "item_id": r["id"], "item_type": "response",
                        "text": f"[Request] {q}\n\n[Response] {r['text']}"})
    return out


def _almost_valid(g: dict, req: str) -> set[str] | None:
    """Itens disputed/guard_error de grupo que seria válido se eles fossem resolvidos a favor."""
    fails = failures(g, req)
    if not fails:
        return None
    ids = set()
    for f in fails:
        if f == "q_harm:intent" and g["q_harm_labels"]["final"] in UNRESOLVED:
            ids.add(f"{g['group_id']}_q_harm")
        elif f == "q_border:intent" and g["q_border_labels"]["final"] in UNRESOLVED:
            ids.add(f"{g['group_id']}_q_border")
        elif not f.startswith("q_"):
            rid = f.split(":")[0]
            r = next(x for x in g["responses"] if x["id"] == rid)
            if r["labels"]["final"] not in UNRESOLVED:
                return None
            ids.add(rid)
        else:
            return None
    return ids


def export(cfg):
    d = data_dir()
    ac = cfg["audit"]
    groups = read_jsonl(d / "groups_p6.jsonl")
    req = cfg["validity"]["required_items"]
    rng = random.Random(cfg["seed"])
    doms = cfg["domain_priority"]
    per = {dom: ac["n_groups"] // len(doms) + 1 for dom in doms}                     # 13 por domínio
    for dom in rng.sample(doms, sum(per.values()) - ac["n_groups"]):                   # um deles com 12
        per[dom] -= 1
    sampled = []
    for dom in doms:
        valid = sorted((g for g in groups if g["valid"] and g["domain"] == dom), key=lambda g: g["group_id"])
        sampled += rng.sample(valid, min(per[dom], len(valid)))
    units = [(g, None) for g in sampled]
    for g in groups:
        ids = None if g["valid"] else _almost_valid(g, req)
        if ids:
            units.append((g, ids))
    rot = ac["pair_rotation"]
    rows = defaultdict(list)
    assign = {}
    for i, (g, ids) in enumerate(units):
        pair = rot[i % len(rot)]
        key = g["group_id"] + ("" if ids is None else ":disputed")
        assign[key] = {"pair": pair, "items": sorted(ids) if ids else "all"}
        for it in _items(g, ids):
            for x in pair:
                rows[x].append(it | {k: "" for k in COLS if k not in it})
    (d / "audit").mkdir(exist_ok=True)
    for x, name in ac["annotators"].items():
        with open(d / "audit" / f"annotator_{x}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, COLS)
            w.writeheader()
            w.writerows(rows[x])
        log(f"audit/annotator_{x}.csv ({name}): {len(rows[x])} linhas")
    write_json(d / "audit" / "assignment.json", {"sampled_groups": [g["group_id"] for g in sampled],
                                                 "n_disputed_units": len(units) - len(sampled), "units": assign})


def _read_annotations(d, ac) -> dict:
    """item_id → {anotador: {campo: valor}}"""
    ann = defaultdict(dict)
    for x in ac["annotators"]:
        p = d / "audit" / f"annotator_{x}.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            vals = {k: (r.get(k) or "").strip().lower() or None for k in ("intent", "class", "coherent", "pair_valid")}
            if any(vals.values()):
                ann[r["item_id"]][x] = vals | {"group_id": r["group_id"], "item_type": r["item_type"], "text": r["text"]}
    return ann


def metrics(cfg):
    d = data_dir()
    ac = cfg["audit"]
    ann = _read_annotations(d, ac)
    res = {"alpha": {}, "n_items": len(ann)}
    for field in ("intent", "class", "coherent", "pair_valid"):
        units = [[v.get(field) for v in by.values()] for by in ann.values()]
        units = [u for u in units if sum(x is not None for x in u) >= 2]
        res["alpha"][field] = krippendorff_alpha_nominal(units) if units else None
    # desacordos → adjudicação por quem não anotou (ordem alfabética dos nomes)
    names = ac["annotators"]
    adj = defaultdict(list)
    for iid, by in ann.items():
        field = "intent" if next(iter(by.values()))["item_type"] == "prompt" else "class"
        vals = {x: v.get(field) for x, v in by.items()}
        if len(by) == 2 and len(set(vals.values())) > 1:
            others = sorted((x for x in names if x not in by), key=lambda x: names[x])
            adj[others[0]].append({"item_id": iid, "group_id": next(iter(by.values()))["group_id"], "field": field,
                                   "text": next(iter(by.values()))["text"],
                                   "options": json.dumps(vals), "label_final": ""})
    for x, rows in adj.items():
        with open(d / "audit" / f"adjudication_{x}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, ["item_id", "group_id", "field", "text", "options", "label_final"])
            w.writeheader()
            w.writerows(rows)
    res["disagreements"] = {x: len(r) for x, r in adj.items()}
    pv = [v["pair_valid"] for by in ann.values() for v in by.values() if v.get("pair_valid")]
    res["pair_valid_rate"] = (sum(p in ("yes", "true", "1", "sim") for p in pv) / len(pv)) if pv else None
    # κ humano (adjudicado/acordo) × consolidado
    human = _human_labels(d, ac, ann)
    groups = {g["group_id"]: g for g in read_jsonl(d / "groups_p6.jsonl")}
    pairs = []
    for iid, h in human.items():
        g = groups.get(gid_of(iid))
        lab = _judge_label(g, iid) if g else None
        if lab and lab not in UNRESOLVED:
            pairs.append((h, lab))
    res["kappa_human_judge"] = cohen_kappa([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None
    res["n_kappa_pairs"] = len(pairs)
    write_json(d / "audit" / "metrics.json", res)
    log(f"P7 metrics {json.dumps(res)}")


def _judge_label(g, iid):
    if iid.endswith("_q_harm"):
        return g["q_harm_labels"]["final"]
    if iid.endswith("_q_border"):
        return g["q_border_labels"]["final"]
    r = next((x for x in g["responses"] if x["id"] == iid), None)
    return r["labels"]["final"] if r else None


def _human_labels(d, ac, ann) -> dict:
    adj = {}
    for x in ac["annotators"]:
        p = d / "audit" / f"adjudication_{x}.csv"
        if p.exists():
            for r in csv.DictReader(open(p)):
                if (r.get("label_final") or "").strip():
                    adj[r["item_id"]] = r["label_final"].strip().lower()
    out = {}
    for iid, by in ann.items():
        field = "intent" if next(iter(by.values()))["item_type"] == "prompt" else "class"
        vals = {v.get(field) for v in by.values()} - {None}
        if iid in adj:
            out[iid] = adj[iid]
        elif len(vals) == 1 and len(by) == 2:
            out[iid] = vals.pop()
    return out


def apply(cfg):
    d = data_dir()
    ann = _read_annotations(d, cfg["audit"])
    human = _human_labels(d, cfg["audit"], ann)
    audited = {gid_of(iid) for iid in ann}
    groups = read_jsonl(d / "groups_p6.jsonl")
    req = cfg["validity"]["required_items"]
    for g in groups:
        g["audited"] = g["group_id"] in audited
        for key, sfx in (("q_harm_labels", "_q_harm"), ("q_border_labels", "_q_border")):
            g[key]["human"] = human.get(g["group_id"] + sfx)
        for r in g["responses"]:
            r["labels"]["human"] = human.get(r["id"])
        f = failures(g, req)
        g["valid"], g["invalid_reason"] = not f, f or None
    write_jsonl(d / "groups_p7.jsonl", groups)
    log(f"P7 apply: {len(human)} rótulos humanos; válidos={sum(g['valid'] for g in groups)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "metrics", "apply"])
    a = ap.parse_args()
    cfg = load_config()
    {"export": export, "metrics": metrics, "apply": apply}[a.cmd](cfg)


if __name__ == "__main__":
    main()
