"""Validador automático do 2×2 por frase-molde (frente D, D-U2). Substitui a auditoria humana.

É uma **segunda checagem, independente**, do que a frente A já valida em `t2_cells.structural_check`.
Aceita os dois formatos de grupo:

- o da frente A (`t2_cells.py`, o que de fato sai no `cells_2x2.jsonl`): `frame` com `{FORM}`/`{INTENT}`,
  `form_alarm`, `form_neutral`, `intent_harmful`, `intent_benign` e `cells[*].slots = {"form", "intent"}`;
- o da seção A1 de `docs/tarefas/gustavo-frente-a.md`: `template` com `{FORMA}`/`{INTENCAO}`, `slots`
  e `cells[*].spans = {"forma", "intencao"}`.

Uso:

    from featgrpo.data.phase1.validate_template import render_cells, validate_group
    erros = validate_group(grupo)           # [] = aprovado; senão, códigos de erro

    PYTHONPATH=src python -m featgrpo.data.phase1.validate_template \
        datasets/phase1/cells_2x2.jsonl --report results/frente-a/validacao_2x2.json

Só checa o que dá para checar por código: estrutura, palavra alarmante presente/ausente, diff entre
células, tamanho e posições. Se a intenção está certa e é a mesma, quem decide são os juízes da frente A.

Regras alinhadas com a frente A (parecer de 2026-11-20 e `results/DESVIOS.md`):
- a presença/ausência é checada pela **palavra alarmante** (`form_alarm`); nos gatilhos `surface`/`action`
  ela precisa ser o gatilho do grupo, e nos `intent_marker` é outra palavra alarmante do Q⁺;
- tamanho: máx/mín de palavras ≤ 1,2 **ou** diferença ≤ 3 palavras;
- se o molde começa por uma lacuna, a primeira letra do texto vai em maiúscula.
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
LEN_RATIO = 1.20
LEN_SLACK = 3
# nomes da frente A -> nomes usados aqui
TYPE_ALIASES = {"surface": "superficie", "action": "acao", "intent_marker": "marcador_intencao"}
FRONT_A_SLOTS = {"forma_trig": "form_alarm", "forma_neutra": "form_neutral",
                 "intencao_nociva": "intent_harmful", "intencao_benigna": "intent_benign"}

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
    text, spans, cursor = "", {}, 0
    for start, ph, value in pos:
        text += template[cursor:start]
        spans["forma" if ph == FORMA else "intencao"] = [len(text), len(text) + len(value)]
        text += value
        cursor = start + len(ph)
    text += template[cursor:]
    if text and text[0].islower():  # mesma regra da frente A
        text = text[0].upper() + text[1:]
    return text, spans


def render_cells(template: str, slots: dict) -> dict:
    """As 4 células do grupo: {célula: {"text": ..., "spans": {"forma": [s, e], "intencao": [s, e]}}}."""
    cells = {}
    for cell, (k_forma, k_int) in CELL_SLOTS.items():
        text, spans = render(template, slots[k_forma], slots[k_int])
        cells[cell] = {"text": text, "spans": spans}
    return cells


def normalize_group(g: dict) -> dict:
    """Converte o formato da frente A para o deste módulo (sem alterar o original)."""
    g = dict(g)
    if g.get("design") == "swap" or ("frame" in g and g.get("frame") is None):
        # plano B da frente A: sem molde; cada linha troca só o trecho do gatilho
        g["template"] = None
        g["row_pieces"] = {
            "harm": (g.get("form_alarm"), g.get("form_neutral")),
            "ben": (g.get("form_alarm_ben") or g.get("form_alarm"), g.get("form_neutral_ben") or g.get("form_neutral")),
        }
        cells = {}
        for c, v in (g.get("cells") or {}).items():
            sl = v.get("spans") or v.get("slots") or {}
            cells[c] = {**v, "spans": {"forma": sl.get("forma", sl.get("form"))}}
        g["cells"] = cells
    elif "frame" in g:
        frame = g.get("frame")
        g["template"] = frame.replace("{FORM}", FORMA).replace("{INTENT}", INTENCAO) if frame else None
        g["slots"] = {k: g.get(v) for k, v in FRONT_A_SLOTS.items()}
        cells = {}
        for c, v in (g.get("cells") or {}).items():
            sl = v.get("spans") or v.get("slots") or {}
            cells[c] = {**v, "spans": {"forma": sl.get("forma", sl.get("form")),
                                       "intencao": sl.get("intencao", sl.get("intent"))}}
        g["cells"] = cells
    g["trigger_type"] = TYPE_ALIASES.get(g.get("trigger_type"), g.get("trigger_type"))
    return g


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


def length_ok(lens: list[int], ratio: float = LEN_RATIO, slack: int = LEN_SLACK) -> bool:
    """Máx/mín ≤ ratio, ou diferença absoluta ≤ slack palavras (regra da frente A)."""
    return min(lens) > 0 and (max(lens) / min(lens) <= ratio or max(lens) - min(lens) <= slack)


def validate_group(g: dict) -> list[str]:
    """Lista de erros do grupo (`código: detalhe`). Lista vazia = aprovado. Aceita os dois formatos."""
    g = normalize_group(g)
    errs: list[str] = []
    for key in ("group_id", "domain", "split", "trigger", "cells"):
        if not g.get(key):
            errs.append(f"campo_ausente: {key}")
    if errs:
        return errs
    if g["split"] not in SPLITS:
        errs.append(f"split_invalido: {g['split']}")

    cells = g["cells"]
    missing = [c for c in CELLS if c not in cells or not (cells[c] or {}).get("text")]
    if missing:
        return errs + [f"celula_ausente: {','.join(missing)}"]
    texts = {c: cells[c]["text"] for c in CELLS}

    if len(set(texts.values())) < 4:
        errs.append("celulas_repetidas")

    template = g.get("template")
    if template is None:
        # plano B: as linhas nociva e benigna vêm de pedidos diferentes; tudo é checado dentro de cada linha
        pieces = g.get("row_pieces") or {"harm": (g["trigger"], None), "ben": (g["trigger"], None)}
        return errs + _validate_swap(texts, cells, pieces)

    errs += _validate_template(g, template, cells)
    alarm = (g.get("slots") or {}).get("forma_trig") or g["trigger"]
    for c in CELLS:
        has = trigger_present(texts[c], alarm)
        if c.endswith("_trig") and not has:
            errs.append(f"gatilho_ausente: {c}")
        if c.endswith("_notrig") and has:
            errs.append(f"gatilho_presente: {c}")
    lens = [_n_words(texts[c]) for c in CELLS]
    if not length_ok(lens):
        errs.append(f"tamanho: {min(lens)}–{max(lens)} palavras")
    return errs


def _validate_template(g: dict, template: str, cells: dict) -> list[str]:
    errs = []
    if template.count(FORMA) != 1 or template.count(INTENCAO) != 1:
        return ["molde_invalido: precisa de uma lacuna de forma e uma de intenção, uma vez cada"]
    slots = g.get("slots") or {}
    absent = [k for k in SLOT_KEYS if not str(slots.get(k) or "").strip()]
    if absent:
        return [f"peca_ausente: {','.join(absent)}"]
    alarm, neutral = slots["forma_trig"], slots["forma_neutra"]
    rest = template.replace(FORMA, " ").replace(INTENCAO, " ")

    if trigger_present(rest, alarm):
        errs.append("gatilho_no_molde: a palavra alarmante aparece fora da lacuna de forma")
    for k in ("intencao_nociva", "intencao_benigna"):
        if trigger_present(slots[k], alarm):
            errs.append(f"gatilho_na_intencao: {k}")
    if set(_stems(alarm)) & set(_stems(neutral)) or alarm.strip().lower() == neutral.strip().lower():
        errs.append("forma_neutra_compartilha_radical")
    if g.get("trigger_type") in ("superficie", "acao"):
        trig = g["trigger"]
        if not (trigger_present(alarm, trig) or trigger_present(trig, alarm)):
            errs.append("forma_alarmante_nao_e_o_gatilho")
    if slots["intencao_nociva"].strip().lower() == slots["intencao_benigna"].strip().lower():
        errs.append("intencoes_iguais")

    expected = render_cells(template, slots)
    for c in CELLS:
        if cells[c]["text"] != expected[c]["text"]:
            errs.append(f"texto_difere_do_molde: {c}")
        elif cells[c].get("spans") != expected[c]["spans"]:
            errs.append(f"spans_errados: {c}")
    return errs


def _expand_to_words(text: str, s: int, e: int) -> tuple[int, int]:
    """Estende [s, e) até as bordas das palavras (corrige recortes como "kill" → "kick" = "ll")."""
    while s > 0 and text[s - 1].isalnum():
        s -= 1
    while e < len(text) and text[e].isalnum():
        e += 1
    return s, e


def _valid_span(span, text: str) -> bool:
    return isinstance(span, (list, tuple)) and len(span) == 2 and 0 <= span[0] < span[1] <= len(text)


def _validate_swap(texts: dict, cells: dict, pieces: dict) -> list[str]:
    """Plano B (`design: "swap"`), linha a linha (nociva e benigna separadas):

    1. o `slots.form` de cada célula é uma posição válida e contém a peça da linha
       (`form_alarm*` em *_trig, `form_neutral*` em *_notrig), sem diferenciar maiúsculas;
    2. **fora** do `slots.form`, as duas versões da linha são idênticas (a troca ficou só ali);
    3. a edição, estendida até a borda das palavras, cabe no `slots.form` (o termo neutro pode manter
       palavras inocentes do gatilho, como "LGBT sickness" → "LGBT illness");
    4. a forma alarmante não aparece na versão sem gatilho;
    5. tamanho comparado **dentro** da linha (entre linhas, os pedidos são diferentes de propósito).
    """
    errs = []
    for row in ("harm", "ben"):
        ct, cn = f"{row}_trig", f"{row}_notrig"
        a, b = texts[ct], texts[cn]
        st, sn = (cells[ct].get("spans") or {}).get("forma"), (cells[cn].get("spans") or {}).get("forma")
        if not (_valid_span(st, a) and _valid_span(sn, b)):
            errs.append(f"spans_errados: {row}")
            continue
        alarm, neutral = pieces.get(row, (None, None))
        if alarm and a[st[0]:st[1]].lower() != alarm.lower():
            errs.append(f"forma_alarmante_fora_do_span: {row}")
        if neutral and b[sn[0]:sn[1]].lower() != neutral.lower():
            errs.append(f"forma_neutra_fora_do_span: {row}")
        if a[:st[0]] != b[:sn[0]] or a[st[1]:] != b[sn[1]:]:
            errs.append(f"texto_muda_fora_da_forma: {row}")
        ea, eb = _single_edit(a, b)
        ea, eb = _expand_to_words(a, *ea), _expand_to_words(b, *eb)
        if ea[0] == ea[1] and eb[0] == eb[1]:
            errs.append(f"sem_edicao: {row}")
        elif not (st[0] <= ea[0] and ea[1] <= st[1] and sn[0] <= eb[0] and eb[1] <= sn[1]):
            errs.append(f"edicao_fora_da_forma: {row}")
        if alarm and trigger_present(b, alarm):
            errs.append(f"gatilho_presente: {cn}")
        if not length_ok([_n_words(a), _n_words(b)]):
            errs.append(f"tamanho: {row} ({_n_words(a)} × {_n_words(b)} palavras)")
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
            ttype = TYPE_ALIASES.get(g.get("trigger_type"), g.get("trigger_type"))
            by_domain[g.get("domain")][0] += 1
            by_domain[g.get("domain")][1] += passed
            by_type[ttype][0] += 1
            by_type[ttype][1] += passed
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
