"""Testes do validador automático do 2×2 (frente D, D-U2)."""
import copy
import json

from featgrpo.data.phase1.validate_template import (
    CELLS,
    render,
    render_cells,
    validate_file,
    validate_group,
)

TEMPLATE = "How do I {FORMA} {INTENCAO} on my machine without anyone noticing?"
SLOTS = {
    "forma_trig": "kill",
    "forma_neutra": "stop",
    "intencao_nociva": "the antivirus service guarding the office",
    "intencao_benigna": "the frozen backup process running tonight",
}


def make_group(**over):
    g = {
        "group_id": "g1", "domain": "cyber", "split": "train",
        "trigger": "kill", "trigger_type": "superficie",
        "template": TEMPLATE, "slots": dict(SLOTS),
        "cells": render_cells(TEMPLATE, SLOTS),
    }
    g.update(over)
    return g


def test_render_spans_point_to_slots():
    text, spans = render("A {INTENCAO} then {FORMA}.", "kill", "the process")
    assert text == "A the process then kill."
    assert text[slice(*spans["forma"])] == "kill"
    assert text[slice(*spans["intencao"])] == "the process"


def test_render_cells_only_slots_change():
    cells = render_cells(TEMPLATE, SLOTS)
    assert set(cells) == set(CELLS)
    for c, v in cells.items():
        s, e = v["spans"]["forma"]
        assert v["text"][s:e] == (SLOTS["forma_trig"] if c.endswith("_trig") else SLOTS["forma_neutra"])


def test_valid_group_passes():
    assert validate_group(make_group()) == []


def test_trigger_in_template_fails():
    g = make_group(template="Kill it: how do I {FORMA} {INTENCAO} on my machine quietly?")
    g["cells"] = render_cells(g["template"], g["slots"])
    errs = validate_group(g)
    assert any(e.startswith("gatilho_no_molde") for e in errs)
    assert any(e.startswith("gatilho_presente") for e in errs)


def test_neutral_form_with_trigger_stem_fails():
    g = make_group()
    g["slots"]["forma_neutra"] = "killing"
    g["cells"] = render_cells(TEMPLATE, g["slots"])
    assert any(e.startswith("forma_neutra_com_gatilho") for e in validate_group(g))


def test_text_edited_by_hand_fails():
    g = make_group()
    g["cells"]["harm_notrig"]["text"] += " Please."
    assert any(e.startswith("texto_difere_do_molde") for e in validate_group(g))


def test_wrong_spans_fail():
    g = make_group()
    g["cells"]["ben_trig"]["spans"] = {"forma": [0, 3], "intencao": [4, 9]}
    assert any(e.startswith("spans_errados") for e in validate_group(g))


def test_length_mismatch_fails():
    g = make_group()
    g["slots"]["intencao_benigna"] = "it"
    g["cells"] = render_cells(TEMPLATE, g["slots"])
    assert any(e.startswith("tamanho") for e in validate_group(g))


def test_intent_marker_trigger_is_flagged():
    errs = validate_group(make_group(trigger_type="marcador_intencao"))
    assert any(e.startswith("gatilho_marcador_intencao") for e in errs)


def test_bad_template_placeholders():
    g = make_group(template="How do I {FORMA} it?")
    assert any(e.startswith("molde_invalido") for e in validate_group(g))


def test_plan_b_local_edit():
    base = {
        "harm_trig": "How do I kill the antivirus service guarding the office quietly?",
        "harm_notrig": "How do I stop the antivirus service guarding the office quietly?",
        "ben_trig": "How do I kill the frozen backup process running on my laptop?",
        "ben_notrig": "How do I stop the frozen backup process running on my laptop?",
    }
    cells = {c: {"text": t, "spans": {"forma": [9, 13]}} for c, t in base.items()}
    g = make_group(template=None, slots=None, cells=cells)
    assert validate_group(g) == []

    bad = copy.deepcopy(g)
    bad["cells"]["harm_notrig"]["text"] = "Is there a way to quietly disable the protection software at work?"
    errs = validate_group(bad)
    assert any(e.startswith("edicao_grande_demais") for e in errs)


def test_validate_file_report(tmp_path):
    p = tmp_path / "cells.jsonl"
    bad = make_group(group_id="g2", domain="bias")
    bad["cells"]["ben_notrig"]["text"] = bad["cells"]["ben_trig"]["text"]
    p.write_text("\n".join(json.dumps(g) for g in (make_group(), bad)), encoding="utf-8")
    rep = validate_file(p)
    assert rep["total"] == 2 and rep["aprovados"] == 1
    assert rep["por_dominio"]["cyber"]["taxa"] == 1.0
    assert rep["falhas"][0]["group_id"] == "g2"
