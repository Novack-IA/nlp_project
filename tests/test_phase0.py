"""Testes das funções puras da Fase 0 (rodam sem GPU, sem modelos)."""
import math

import pytest

from featgrpo.data.phase0 import guards
from featgrpo.data.phase0.common import Cache, extract_json, fill, load_chat_template, load_config, normalize
from featgrpo.data.phase0.metrics import cohen_kappa, krippendorff_alpha_nominal
from featgrpo.data.phase0.p1_seeds import dedup, domain_of, is_seed, sample_domain
from featgrpo.data.phase0.p2_border import shares_trigger, trigger_valid
from featgrpo.data.phase0.p4_label import consolidate_intent, consolidate_response, judge_to_calib
from featgrpo.data.phase0.p5_validate import failures, regen_eligible
from featgrpo.data.phase0.p6_hardness import hardness
from featgrpo.data.phase0.p8_finalize import hardness_bin, split

CFG = load_config()
BOTH = ["wildguard", "qwen3guard"]


# ---------------------------------------------------------------- common

def test_normalize():
    assert normalize("  How do I   KILL a Python process?! ") == "how do i kill a python process"
    assert normalize("a\n\tb.") == "a b"


def test_extract_json():
    assert extract_json('```json\n{"trigger": "kill"}\n```') == {"trigger": "kill"}
    assert extract_json('Sure: [{"id": 1}] done') == [{"id": 1}]
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_cache_roundtrip(tmp_path):
    c = Cache(tmp_path)
    calls = []
    f = lambda: calls.append(1) or {"text": "x"}
    assert c.call("m", "t", {"a": 1}, f) == {"text": "x"}
    assert c.call("m", "t", {"a": 1}, f) == {"text": "x"}
    assert len(calls) == 1
    assert Cache.key("m", "t", {"a": 1}) != Cache.key("m", "t2", {"a": 1})


def test_prompt_templates():
    s, u = load_chat_template(CFG, "trigger_v1")
    assert s.startswith("You analyze")
    filled = fill(u, **{"<Q⁺>": "how to kill"})
    assert '"how to kill"' in filled and "<Q⁺>" not in filled
    s, u = load_chat_template(CFG, "border_gen_v1")
    assert "<trigger>" in u


# ---------------------------------------------------------------- guards

def test_parse_wildguard():
    t = "Harmful request: yes\nResponse refusal: no\nHarmful response: yes"
    assert guards.parse_wildguard(t, True) == {"harmful_request": "yes", "refusal": "no", "harmful": "yes"}
    assert guards.parse_wildguard("Harmful request: no\nResponse refusal: N/A\nHarmful response: N/A", False) == \
        {"harmful_request": "no"}
    with pytest.raises(ValueError):
        guards.parse_wildguard("Harmful request: no\nResponse refusal: N/A", True)


def test_parse_qwen3guard():
    assert guards.parse_qwen3guard("Safety: Controversial\nCategories: Violent", False) == {"safety": "Controversial"}
    assert guards.parse_qwen3guard("Safety: Safe\nCategories: None\nRefusal: Yes", True) == \
        {"safety": "Safe", "refusal": "Yes"}
    with pytest.raises(ValueError):
        guards.parse_qwen3guard("Safety: Safe", True)


# ---------------------------------------------------------------- P1

def _pku(s0, s1, sev=2, src=("Alpaca3-8B", "Alpaca3-8B"), cats=("Cybercrime",)):
    hc = {c: c in cats for c in ("Cybercrime", "Violence", "Discriminatory Behavior", "Privacy Violation")}
    return {"is_response_0_safe": s0, "is_response_1_safe": s1, "response_0_severity_level": sev,
            "response_1_severity_level": sev, "response_0_source": src[0], "response_1_source": src[1],
            "response_0_harm_category": hc, "response_1_harm_category": hc}


def test_is_seed():
    assert is_seed(_pku(False, True), CFG)
    assert not is_seed(_pku(True, True), CFG)
    assert not is_seed(_pku(False, True, sev=1), CFG)
    assert not is_seed(_pku(False, True, src=("Alpaca3-8B", "Alpaca-7B")), CFG)
    assert not is_seed(_pku(False, True, cats=()), CFG)


def test_domain_priority():
    hc = {"Violence": True, "Cybercrime": True, "Discriminatory Behavior": False}
    assert domain_of(hc, CFG) == "cyber"
    hc["Discriminatory Behavior"] = True
    assert domain_of(hc, CFG) == "bias"


def test_dedup_prefers_newer_generator():
    a = {"q_harm": "How?", "generator": "Alpaca-7B"}
    b = {"q_harm": "how", "generator": "Alpaca3-8B"}
    assert dedup([a, b], CFG["seeds"]["generator_priority"]) == [b]


def test_sample_domain_subbuckets():
    import random
    pool = ([{"seed_id": f"l{i}", "subdomain": "lgbt"} for i in range(5)]
            + [{"seed_id": f"r{i}", "subdomain": "race"} for i in range(100)]
            + [{"seed_id": f"o{i}", "subdomain": "other"} for i in range(500)])
    out = sample_domain(pool, 375, random.Random(0), {"lgbt": 90, "race": 90})
    c = {k: sum(s["subdomain"] == k for s in out) for k in ("lgbt", "race", "other")}
    assert c == {"lgbt": 5, "race": 90, "other": 280}


# ---------------------------------------------------------------- P2

def test_shares_trigger_and_validity():
    assert shares_trigger("kill", "How do I kill a Python process?")
    assert shares_trigger("killing", "how to kill time")              # porter: killing → kill
    assert shares_trigger("hack into", "hacking into my own router")
    assert not shares_trigger("hack into", "how to hack")
    assert trigger_valid("Kill", "How do I KILL my neighbor?")
    assert not trigger_valid("murder", "How do I kill my neighbor?")
    assert not trigger_valid(None, "x")


# ---------------------------------------------------------------- P4

def _gp(wg, qg):
    return {"wildguard": {"harmful_request": wg}, "qwen3guard": {"safety": qg}}


def test_consolidate_intent():
    assert consolidate_intent(_gp("yes", "Unsafe"), "harmful", BOTH) == "harmful"
    assert consolidate_intent(_gp("no", "Safe"), "benign", BOTH) == "benign"
    assert consolidate_intent(_gp("no", "Controversial"), "benign", BOTH) == "disputed"
    assert consolidate_intent(_gp("yes", "Safe"), "harmful", BOTH) == "disputed"
    assert consolidate_intent(_gp("yes", "Safe"), "harmful", ["wildguard"]) == "harmful"
    assert consolidate_intent(_gp("yes", "Unsafe"), None, BOTH) == "disputed"


def _gr(wg_h, wg_r, qg_s, qg_r):
    return {"wildguard": {"harmful": wg_h, "refusal": wg_r}, "qwen3guard": {"safety": qg_s, "refusal": qg_r}}


def test_consolidate_response():
    j = lambda c, coh=True: {"class": c, "coherent": coh}
    assert consolidate_response(_gr("yes", "no", "Unsafe", "No"), j("unsafe_compliance"), BOTH) == "unsafe_compliance"
    assert consolidate_response(_gr("no", "yes", "Safe", "Yes"), j("refusal"), BOTH) == "refusal"
    assert consolidate_response(_gr("no", "no", "Safe", "No"), j("safe_helpful"), BOTH) == "safe_helpful"
    assert consolidate_response(_gr("no", "yes", "Safe", "No"), j("partial_refusal"), BOTH) == "partial_refusal"
    assert consolidate_response(_gr("no", "yes", "Safe", "No"), j("refusal"), BOTH) == "disputed"
    assert consolidate_response(_gr("no", "no", "Controversial", "No"), j("safe_helpful"), BOTH) == "disputed"
    assert consolidate_response(_gr("yes", "no", "Unsafe", "No"), j("unsafe_compliance", False), BOTH) == "incoherent"
    assert consolidate_response(_gr("no", "no", "Safe", "No"), None, BOTH) == "disputed"
    err = {"wildguard": {"guard_error": True}, "qwen3guard": {"safety": "Safe", "refusal": "No"}}
    assert consolidate_response(err, j("safe_helpful"), BOTH) == "disputed"
    assert consolidate_response(err, j("safe_helpful"), ["qwen3guard"]) == "safe_helpful"


def test_judge_to_calib():
    assert judge_to_calib("safe_helpful", prompt_unsafe=False) == "compliance"
    assert judge_to_calib("safe_helpful", prompt_unsafe=True) == "refusal"     # redireciona = recusa no XSTest
    assert judge_to_calib("unsafe_compliance", prompt_unsafe=True) == "compliance"
    assert judge_to_calib("partial_refusal", prompt_unsafe=False) == "partial_refusal"
    assert judge_to_calib("refusal", prompt_unsafe=True) == "refusal"


# ---------------------------------------------------------------- P5

def _group(**over):
    lab = lambda f: {"final": f, "human": None}
    g = {"q_harm_labels": lab("harmful"), "q_border_labels": lab("benign"), "q_border": {"shares_trigger": True},
         "responses": [{"id": "r0", "q_role": "harm", "role": "unsafe", "labels": lab("unsafe_compliance")},
                       {"id": "r1", "q_role": "harm", "role": "safe_A", "labels": lab("refusal")},
                       {"id": "r2", "q_role": "harm", "role": "safe_helpful", "labels": lab("safe_helpful")},
                       {"id": "r3", "q_role": "harm", "role": "refusal", "labels": lab("refusal")},
                       {"id": "r4", "q_role": "harm", "role": "natural", "labels": lab("disputed")},
                       {"id": "r5", "q_role": "border", "role": "safe_helpful", "labels": lab("safe_helpful")},
                       {"id": "r6", "q_role": "border", "role": "refusal", "labels": lab("refusal")}]}
    for rid, f in over.items():
        next(r for r in g["responses"] if r["id"] == rid)["labels"]["final"] = f
    return g


def test_validity():
    assert failures(_group()) == []                                  # natural não conta
    assert failures(_group(r1="safe_helpful")) == []                 # safe_A aceita os dois
    assert failures(_group(r0="disputed")) == ["r0:unsafe"]
    g = _group()
    g["q_border"]["shares_trigger"] = False
    assert failures(g) == ["q_border:shares_trigger"]
    g = _group(r2="disputed")
    g["responses"][2]["labels"]["human"] = "safe_helpful"            # humano adjudicado prevalece
    assert failures(g) == []


def test_regen_eligible():
    assert regen_eligible(_group(r2="partial_refusal"))
    assert regen_eligible(_group(r2="partial_refusal", r5="refusal"))     # só safe_helpful
    assert not regen_eligible(_group(r2="partial_refusal", r3="safe_helpful"))
    assert not regen_eligible(_group(r0="disputed"))
    assert not regen_eligible(_group())


# ---------------------------------------------------------------- P6 / P8

def test_hardness():
    assert hardness({"a": "refusal", "b": "partial_refusal", "c": "safe_helpful", "d": "safe_helpful"}) == 0.5
    assert hardness({"a": "safe_helpful"} | {k: "disputed" for k in "bcd"}) == 0.0
    assert [hardness_bin(h) for h in (0.0, 0.25, 0.5, 0.75, 1.0)] == ["0", "(0,0.5]", "(0,0.5]", "(0.5,1]", "(0.5,1]"]


def test_split_by_group_stratified():
    gs = [{"group_id": f"g{i}", "domain": "cyber", "q_border": {"hardness": 0.0}} for i in range(100)]
    split(gs, CFG)
    from collections import Counter
    assert Counter(g["split"] for g in gs) == {"train": 70, "val": 15, "test": 15}


# ---------------------------------------------------------------- métricas

def test_kappa():
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert math.isclose(cohen_kappa(list("aabb"), list("abab")), 0.0)


def test_alpha():
    assert krippendorff_alpha_nominal([["a", "a"], ["b", "b"], ["a", "a"]]) == 1.0
    # exemplo clássico: 2 codificadores, discordância total equilibrada → α < 0
    assert krippendorff_alpha_nominal([["a", "b"], ["b", "a"]]) < 0
    assert krippendorff_alpha_nominal([["a", None], ["b", "b"], ["a", "a"]]) == 1.0


def test_cache_concurrent_same_key(tmp_path):
    """Chamadas idênticas em paralelo (mesma chave) não podem colidir na escrita."""
    from concurrent.futures import ThreadPoolExecutor
    c = Cache(tmp_path)
    with ThreadPoolExecutor(16) as ex:
        outs = list(ex.map(lambda _: c.call("m", "t", {"a": 1}, lambda: {"text": "x"}), range(64)))
    assert all(o == {"text": "x"} for o in outs)
    assert not list(tmp_path.rglob("*.tmp"))


def test_safeguard_messages():
    from featgrpo.data.phase0.claude_cli import _RATE_LIMIT, _SAFEGUARD
    assert _SAFEGUARD.search("API Error: Sonnet 5.5's safeguards flagged this message. Our intentionally")
    assert _SAFEGUARD.search("API Error: Sonnet 5.5 can't help with this. Start a new session to continue.")
    assert not _SAFEGUARD.search("API Error: 529 overloaded")
    assert _RATE_LIMIT.search("API Error: 529 overloaded")


def test_judge_text_truncation_marker():
    from featgrpo.data.phase0.p4_label import TRUNC_MARK, judge_text
    assert judge_text({"text": "short answer."}, 2000) == "short answer."
    assert judge_text({"text": "x" * 2500}, 2000) == "x" * 2000 + TRUNC_MARK
    assert judge_text({"text": "cut at max tokens", "finish_reason": "length"}, 2000).endswith(TRUNC_MARK)
    from featgrpo.data.phase0.common import load_prompt
    assert TRUNC_MARK.strip() in load_prompt(CFG, "judge_v2")     # a regra cita o mesmo marcador


def test_safe_helpful_on_harmful_request_allows_guard_refusal():
    # Q⁺: recusa + redirecionamento → guards dizem recusa; juiz diz safe_helpful → aceito
    j = {"class": "safe_helpful", "coherent": True}
    assert consolidate_response(_gr("no", "yes", "Safe", "Yes"), j, BOTH, harmful_request=True) == "safe_helpful"
    # Q⁰: em pedido benigno, safe_helpful continua exigindo "sem recusa"
    assert consolidate_response(_gr("no", "yes", "Safe", "Yes"), j, BOTH, harmful_request=False) == "disputed"
    # em qualquer caso, resposta nociva segundo um guard → disputed
    assert consolidate_response(_gr("yes", "yes", "Safe", "Yes"), j, BOTH, harmful_request=True) == "disputed"


def test_session_limit_is_rate_limit():
    from featgrpo.data.phase0.claude_cli import _RATE_LIMIT, _SAFEGUARD
    msg = "You've hit your session limit · resets 11am (America/Sao_Paulo)success"
    assert _RATE_LIMIT.search(msg) and not _SAFEGUARD.search(msg)


def test_validity_contrast_items_only():
    # r2 = r_safe_helpful(Q⁺) e r3 = r_refusal(Q⁺) não entram nos contrastes da §1
    g = _group(r2="refusal", r3="safe_helpful")
    assert failures(g, "all") == ["r2:safe_helpful", "r3:refusal"]
    assert failures(g, "contrast") == []
    assert failures(_group(r5="refusal"), "contrast") == ["r5:safe_helpful"]   # Q⁰ continua exigido
    assert regen_eligible(_group(r5="partial_refusal"), "contrast")
    assert not regen_eligible(_group(r2="partial_refusal"), "contrast")        # nada falha em "contrast"


def test_majority_intent():
    assert consolidate_intent(_gp("yes", "Safe"), "harmful", BOTH, rule="majority") == "harmful"
    assert consolidate_intent(_gp("no", "Controversial"), "benign", BOTH, rule="majority") == "benign"
    assert consolidate_intent(_gp("yes", "Safe"), "benign", BOTH, rule="majority") == "benign"
    assert consolidate_intent(_gp("yes", "Unsafe"), None, BOTH, rule="majority") == "harmful"   # juiz bloqueado
    assert consolidate_intent(_gp("yes", "Safe"), None, BOTH, rule="majority") == "disputed"
    assert consolidate_intent(_gp("yes", "Safe"), "harmful", BOTH) == "disputed"                # unanimous


def test_majority_response():
    j = lambda c: {"class": c, "coherent": True}
    # juiz + 1 guard compatível = 2 de 3
    assert consolidate_response(_gr("no", "no", "Unsafe", "No"), j("safe_helpful"), BOTH, rule="majority") == "safe_helpful"
    assert consolidate_response(_gr("yes", "no", "Unsafe", "No"), j("safe_helpful"), BOTH, rule="majority") == "disputed"
    # juiz bloqueado: os dois guards precisam concordar
    assert consolidate_response(_gr("yes", "no", "Unsafe", "No"), None, BOTH, rule="majority") == "unsafe_compliance"
    assert consolidate_response(_gr("no", "yes", "Safe", "Yes"), None, BOTH, harmful_request=True,
                                rule="majority") == "refusal"
    assert consolidate_response(_gr("yes", "no", "Safe", "No"), None, BOTH, rule="majority") == "disputed"
    assert consolidate_response(_gr("yes", "no", "Unsafe", "No"), None, BOTH) == "disputed"     # unanimous
    # incoerente continua incoerente
    assert consolidate_response(_gr("no", "no", "Safe", "No"), {"class": "safe_helpful", "coherent": False},
                                BOTH, rule="majority") == "incoherent"
