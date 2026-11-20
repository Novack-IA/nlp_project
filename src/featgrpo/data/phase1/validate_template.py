"""Validador automático do 2×2 por frase-molde (frente D, D-U2). Substitui a auditoria humana.

A frente A usa este módulo de dois jeitos:

    from featgrpo.data.phase1.validate_template import render_cells, validate_group

    cells = render_cells(template, slots)   # monta as 4 células + posições das lacunas
    erros = validate_group(grupo)           # [] = grupo aprovado; senão, códigos de erro

E na linha de comando, sobre o arquivo inteiro:

    PYTHONPATH=src python -m featgrpo.data.phase1.validate_template \
        datasets/phase1/cells_2x2.jsonl --report results/frente-a/validacao_2x2.json

O formato do grupo está em `docs/tarefas/gustavo-frente-a.md`, seção A1. Este módulo só
checa o que dá para checar por código: estrutura, gatilho, diff entre células, tamanho e
posições. Se a intenção está certa e é a mesma do original, quem decide são os juízes da frente A.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from nltk.stem import PorterStemmer

FORMA = "{FORMA}"
INTENCAO = "{INTENCAO}"
CELLS = ("harm_trig", "harm_notrig", "ben_trig", "ben_notrig")
# célula -> (peça da lacuna de forma, peça da lacuna de intenção)
CELL_SLOTS = {
    "harm_trig": ("forma_trig", "intencao_nociva"),
    "harm_notrig": ("forma_neutra", "intencao_nociva"),
    "ben_trig": ("forma_trig", "intencao_benigna"),
    "ben_notrig": ("forma_neutra", "intencao_benigna"),
}
SLOT_KEYS = ("forma_trig", "forma_neutra", "intencao_nociva", "intencao_benigna")
SPLITS = ("train", "val", "test")
MAX_LEN_DIFF = 0.20

_STEM = PorterStemmer()
_TOKEN = re.compile(r"[a-z0-9']+")


# ---------------------------------------------------------------- gatilho (compartilhado com interp.baselines)

def _stems(text: str) -> list[str]:
    return [_STEM.stem(t) for t in _TOKEN.findall(text.lower())]


def normalize_trigger(trigger: str) -> str:
    """Forma canônica do gatilho (minúsculas + radical Porter), para agrupar e comparar."""
    return " ".join(_stems(trigger))


def trigger_present(text: str, trigger: str) -> bool:
    """Gatilho no texto: literal (sem caixa) ou, se falhar, todos os radicais presentes."""
    if trigger.lower().strip() in text.lower():
        return True
    trig = set(_stems(trigger))
    return bool(trig) and trig <= set(_stems(text))


# ---------------------------------------------------------------- montagem

def render(template: str, forma: str, intencao: str) -> tuple[str, dict]:
    """Preenche o molde e devolve o texto e as posições [início, fim) de cada lacuna."""
    pos = sorted([(template.index(FORMA), FORMA, forma), (template.index(INTENCAO), INTENCAO, intencao)])
    out, spans, cursor = [], {}, 0
    for start, ph, value in pos:
        out.append(template[cursor:start])
        s = sum(len(p) for p in out)
        out.append(value)
        spans["forma" if ph == FORMA else "intencao"] = [s, s + len(value)]
        cursor = start + len(ph)
    out.append(template[cursor:])
    return "".join(out), spans


def render_cells(template: str, slots: dict) -> dict:
    """As 4 células do grupo: {célula: {"text": ..., "spans": {"forma": [s, e], "intencao": [s, e]}}}."""
    cells = {}
    for cell, (k_forma, k_int) in CELL_SLOTS.items():
        text, spans = render(template, slots[k_forma], slots[k_int])
        cells[cell] = {"text": text, "spans": spans}
    return cells


# ---------------------------------------------------------------- validação

def _single_edit(a: str, b: str) -> tuple[list[int], list[int]]:
    """Trecho que muda de a para b (prefixo e sufixo comuns removidos), em posições de a e de b."""
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and a[-1 - j] == b[-1 - j]:
        j += 1
    return [i, len(a) - j], [i, len(b) - j]


def _n_words(text: str) -> int:
    return len(text.split())


def validate_group(g: dict, max_len_diff: float = MAX_LEN_DIFF) -> list[str]:
    """Lista de erros do grupo (`código: detalhe`). Lista vazia = aprovado."""
    errs: list[str] = []
    for key in ("group_id", "domain", "split", "trigger", "cells"):
        if not g.get(key):
            errs.append(f"campo_ausente: {key}")
    if errs:
        return errs
    if g["split"] not in SPLITS:
        errs.append(f"split_invalido: {g['split']}")
    if g.get("trigger_type") == "marcador_intencao":
        errs.append("gatilho_marcador_intencao: fica fora da análise principal")

    trigger = g["trigger"]
    cells = g["cells"]
    missing = [c for c in CELLS if c not in cells or not (cells[c] or {}).get("text")]
    if missing:
        return errs + [f"celula_ausente: {','.join(missing)}"]
    texts = {c: cells[c]["text"] for c in CELLS}

    template = g.get("template")
    if template is not None:
        errs += _validate_template(g, template, trigger, cells)
    else:
        errs += _validate_local_edit(texts, trigger, cells)

    # gatilho presente/ausente
    for c in CELLS:
        has = trigger_present(texts[c], trigger)
        if c.endswith("_trig") and not has:
            errs.append(f"gatilho_ausente: {c}")
        if c.endswith("_notrig") and has:
            errs.append(f"gatilho_presente: {c}")

    # tamanho
    lens = [_n_words(texts[c]) for c in CELLS]
    if max(lens) and (max(lens) - min(lens)) / max(lens) > max_len_diff:
        errs.append(f"tamanho: {min(lens)}–{max(lens)} palavras (> {max_len_diff:.0%})")

    # textos iguais entre células que deveriam diferir
    if len(set(texts.values())) < 4:
        errs.append("celulas_repetidas")
    return errs


def _validate_template(g: dict, template: str, trigger: str, cells: dict) -> list[str]:
    errs = []
    if template.count(FORMA) != 1 or template.count(INTENCAO) != 1:
        return [f"molde_invalido: precisa de {FORMA} e {INTENCAO} exatamente uma vez cada"]
    slots = g.get("slots") or {}
    absent = [k for k in SLOT_KEYS if not str(slots.get(k, "")).strip()]
    if absent:
        return [f"peca_ausente: {','.join(absent)}"]
    if trigger_present(template.replace(FORMA, " ").replace(INTENCAO, " "), trigger):
        errs.append("gatilho_no_molde: o gatilho aparece fora da lacuna de forma")
    if not trigger_present(slots["forma_trig"], trigger):
        errs.append("forma_trig_sem_gatilho")
    if trigger_present(slots["forma_neutra"], trigger):
        errs.append("forma_neutra_com_gatilho")
    for k in ("intencao_nociva", "intencao_benigna"):
        if trigger_present(slots[k], trigger):
            errs.append(f"gatilho_na_intencao: {k}")
    if slots["forma_trig"].strip().lower() == slots["forma_neutra"].strip().lower():
        errs.append("formas_iguais")
    if slots["intencao_nociva"].strip().lower() == slots["intencao_benigna"].strip().lower():
        errs.append("intencoes_iguais")

    expected = render_cells(template, slots)
    for c in CELLS:
        if cells[c]["text"] != expected[c]["text"]:
            errs.append(f"texto_difere_do_molde: {c}")
        elif cells[c].get("spans") != expected[c]["spans"]:
            errs.append(f"spans_errados: {c}")
    return errs


def _validate_local_edit(texts: dict, trigger: str, cells: dict) -> list[str]:
    """Plano B (sem molde): cada célula *_notrig difere da *_trig só num trecho, que contém o gatilho."""
    errs = []
    for row in ("harm", "ben"):
        a, b = texts[f"{row}_trig"], texts[f"{row}_notrig"]
        sa, _ = _single_edit(a, b)
        changed = a[sa[0]:sa[1]]
        # o trecho trocado precisa cobrir o gatilho, e não pode ser quase a frase toda
        if not trigger_present(changed, trigger):
            errs.append(f"edicao_nao_cobre_gatilho: {row}")
        if len(changed) > 0.5 * len(a):
            errs.append(f"edicao_grande_demais: {row} ({len(changed)}/{len(a)} caracteres)")
    for c in CELLS:
        spans = cells[c].get("spans") or {}
        f = spans.get("forma")
        if f is None or not (0 <= f[0] < f[1] <= len(texts[c])):
            errs.append(f"spans_errados: {c}")
    return errs


# ---------------------------------------------------------------- arquivo inteiro

def validate_file(path: Path) -> dict:
    """Valida todos os grupos de um JSONL e resume os resultados."""
    total, ok = 0, 0
    reasons: Counter = Counter()
    by_domain: dict = defaultdict(lambda: [0, 0])
    by_type: dict = defaultdict(lambda: [0, 0])
    failures = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            total += 1
            errs = validate_group(g)
            passed = not errs
            ok += passed
            by_domain[g.get("domain")][0] += 1
            by_domain[g.get("domain")][1] += passed
            by_type[g.get("trigger_type")][0] += 1
            by_type[g.get("trigger_type")][1] += passed
            for e in errs:
                reasons[e.split(":")[0]] += 1
            if errs:
                failures.append({"group_id": g.get("group_id"), "erros": errs})

    def rate(d):
        return {str(k): {"total": n, "aprovados": a, "taxa": round(a / n, 3) if n else None} for k, (n, a) in d.items()}

    return {
        "arquivo": str(path),
        "total": total,
        "aprovados": ok,
        "taxa_aprovacao": round(ok / total, 3) if total else None,
        "motivos": dict(reasons.most_common()),
        "por_dominio": rate(by_domain),
        "por_tipo_gatilho": rate(by_type),
        "falhas": failures,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Valida o 2×2 por frase-molde.")
    ap.add_argument("path", type=Path)
    ap.add_argument("--report", type=Path, help="onde salvar o relatório JSON")
    args = ap.parse_args(argv)
    rep = validate_file(args.path)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{rep['aprovados']}/{rep['total']} grupos aprovados ({rep['taxa_aprovacao']})")
    for k, v in rep["motivos"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
