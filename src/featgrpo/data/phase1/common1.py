"""Utilidades da Fase 1 (frente A). Reaproveita a infraestrutura da Fase 0 (cache, cliente vLLM, guards).

Diretórios:
  PHASE1_DATA (default data/phase1)  saídas intermediárias (fora do git)
  datasets/phase1/                   entregas (no git)
O cache de chamadas é o mesmo da Fase 0 (PHASE0_CACHE).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from ..phase0.common import REPO, Cache, LLM, extract_json, log

PROMPTS1 = Path(__file__).resolve().parent / "prompts"
OUT = REPO / "datasets" / "phase1"
ACTIVE_GUARDS = ["wildguard", "qwen3guard"]
JUDGE_MODEL = "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
GEN_MODEL = os.environ.get("PHASE1_GEN_MODEL", "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic")
SUBJECT4B = "google/gemma-3-4b-it"
PAIRS = REPO / "datasets" / "phase0" / "v5b" / "pairs_prompt_level.jsonl"


def data1() -> Path:
    d = Path(os.environ.get("PHASE1_DATA", "data/phase1"))
    d = d if d.is_absolute() else REPO / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def chat_template(name: str) -> tuple[str, str]:
    t = (PROMPTS1 / f"{name}.txt").read_text()
    m = re.match(r"SYSTEM:\s*(.*?)\n\s*\nUSER:\s*(.*)\Z", t, re.S)
    if not m:
        raise ValueError(f"template {name} sem SYSTEM/USER")
    return m.group(1).strip(), m.group(2).rstrip("\n")


def fill(t: str, **kw) -> str:
    for k, v in kw.items():
        t = t.replace("<" + k + ">", str(v))
    return t


def hfrac(s: str) -> float:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def judge_json(llm: LLM, cache: Cache, template: str, user: str, schema: dict, validate,
               max_tokens: int = 1024) -> dict | None:
    """Chamada ao juiz com structured output; 2 tentativas; None se inválido."""
    system, _ = chat_template(template)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(2):
        def fn():
            extra = {"response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}}
            return llm.chat(msgs, temperature=0.0 if attempt == 0 else 0.3, max_tokens=max_tokens,
                            seed=attempt, extra=extra)[0]
        raw = cache.call(llm.model, f"p1:{template}#a{attempt}", {"user": user}, fn)
        try:
            obj = extract_json(raw["text"])
            if validate(obj):
                return obj
        except (ValueError, TypeError, KeyError):
            pass
    return None


# ---------------------------------------------------------------- texto

_WORD = re.compile(r"[A-Za-z0-9']+")
_STOP = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "into", "their", "his", "her", "my",
         "your", "with", "without", "from", "by", "at", "as", "is", "be", "it", "its", "that", "this", "someone",
         "someone's", "people", "person", "s"}
_stemmer = None


def stems(s: str) -> set[str]:
    global _stemmer
    if _stemmer is None:
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
    return {_stemmer.stem(w.lower()) for w in _WORD.findall(s or "") if w.lower() not in _STOP}


def contains_phrase(text: str, phrase: str) -> bool:
    """Frase presente: literal (case-insensitive) ou todos os radicais de conteúdo presentes."""
    if phrase.lower() in text.lower():
        return True
    ps = stems(phrase)
    return bool(ps) and ps <= stems(text)


def n_words(s: str) -> int:
    return len(_WORD.findall(s or ""))


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


__all__ = ["data1", "out_dir", "chat_template", "fill", "hfrac", "judge_json", "stems", "contains_phrase",
           "n_words", "dump", "log", "ACTIVE_GUARDS", "JUDGE_MODEL", "GEN_MODEL", "SUBJECT4B", "PAIRS"]
