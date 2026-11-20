"""Monta o pacote de release do dataset da Fase 0 (para o Hugging Face, privado/gated).

Uso:
    PYTHONPATH=src python3 -m featgrpo.data.phase0.export_release \
        --v5b data/phase0_v5b --v5 data/phase0_b200 --out /raid/.../hf_release/phase0

Saída:
    v5b/{train,val,test}.jsonl     dataset final (500 grupos, 125/domínio, 8 respostas por grupo)
    v5b/pool_groups.jsonl          todos os 2.806 grupos da v5b com rótulos (válidos ou não)
    v5b/pairs_prompt_level.jsonl   2.428 pares Q⁺/Q⁰ com pedidos válidos (sem exigir respostas)
    v5b/{manifest.json,report.md,calibration.json,compare_v5_vs_v5b.md,reports/,audit/}
    v5/{train,val,test}.jsonl, v5/pool_groups.jsonl, v5/{manifest.json,report.md}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def read_jsonl(p: Path):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def hash_split(gid: str) -> str:
    h = int(hashlib.sha256(gid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < 0.70 else ("val" if h < 0.85 else "test")


def prompt_valid(g: dict) -> bool:
    return (g["q_harm_labels"]["final"] == "harmful"
            and g["q_border_labels"]["final"] == "benign"
            and bool(g["q_border"].get("shares_trigger")))


def natural_class(g: dict, q_role: str):
    for r in g["responses"]:
        if r["q_role"] == q_role and r["role"] == "natural":
            return (r.get("labels") or {}).get("final")
    return None


def export_version(src: Path, dst: Path, with_extras: bool):
    final = src / "final"
    split_of = {}
    for s in ("train", "val", "test"):
        p = final / f"{s}.jsonl"
        if p.exists():
            rows = read_jsonl(p)
            write_jsonl(dst / f"{s}.jsonl", rows)
            split_of.update({r["group_id"]: s for r in rows})
    for name in ("manifest.json", "report.md"):
        if (final / name).exists():
            shutil.copy2(final / name, dst / name)
    pool = read_jsonl(src / "groups_p6.jsonl")
    write_jsonl(dst / "pool_groups.jsonl", pool)
    if not with_extras:
        return split_of, pool
    pairs = []
    for g in pool:
        if not prompt_valid(g):
            continue
        pairs.append({
            "group_id": g["group_id"],
            "domain": g["domain"],
            "subdomain": g.get("subdomain_judge") or g.get("subdomain"),
            "trigger": g["trigger"],
            "q_harm": g["q_harm"],
            "q_border": g["q_border"]["text"],
            "q_border_method": g["q_border"]["method"],
            "in_final": g["group_id"] in split_of,
            "valid_full_group": bool(g.get("valid")),
            "split": split_of.get(g["group_id"]) or hash_split(g["group_id"]),
            "natural_class_q_harm": natural_class(g, "harm"),
            "natural_class_q_border": natural_class(g, "border"),
        })
    write_jsonl(dst / "pairs_prompt_level.jsonl", pairs)
    for name in ("calibration.json",):
        shutil.copy2(src / name, dst / name)
    rep = dst / "reports"
    rep.mkdir(exist_ok=True)
    for p in sorted(src.glob("*_report.json")):
        shutil.copy2(p, rep / p.name)
    for p in sorted(src.glob("compare_*.md")):
        shutil.copy2(p, dst / "compare_v5_vs_v5b.md")
    if (src / "audit").exists():
        shutil.copytree(src / "audit", dst / "audit", dirs_exist_ok=True)
    return split_of, pool, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5b", default="data/phase0_v5b")
    ap.add_argument("--v5", default="data/phase0_b200")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split_of, pool, pairs = export_version(Path(a.v5b), out / "v5b", with_extras=True)
    export_version(Path(a.v5), out / "v5", with_extras=False)
    from collections import Counter
    print(json.dumps({
        "v5b_final": Counter(split_of.values()),
        "v5b_pool": len(pool),
        "v5b_pairs_prompt_level": len(pairs),
        "pairs_by_split": Counter(p["split"] for p in pairs),
        "pairs_by_domain": Counter(p["domain"] for p in pairs),
    }, indent=1, default=dict))


if __name__ == "__main__":
    main()
