"""Cliente do Claude Code CLI (juiz da v4, Spark). Mantido para reprodutibilidade da v4.

Na phase0-v5-h100 o juiz geral é local (judge.py); este módulo não é usado pelo pipeline.
"""
from __future__ import annotations

import json
import os
import re
import subprocess

# mensagens do CLI: bloqueio pelas salvaguardas do modelo × limite/erro transitório
_SAFEGUARD = re.compile(r"safeguards flagged|can't help with this|cannot help with this", re.I)
_RATE_LIMIT = re.compile(r"\b529\b|overloaded|rate.?limit|session limit|usage limit|hit your", re.I)


def call(prompt: str, system_prompt_file: str, model: str = "claude-sonnet-5-5", workdir: str = ".") -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY definido: o CLI cobraria como API (§5.1)")
    out = subprocess.run(["claude", "-p", prompt, "--model", model, "--system-prompt-file", system_prompt_file,
                          "--output-format", "json"], cwd=workdir, capture_output=True, text=True, timeout=600)
    return json.loads(out.stdout)["result"]
