"""Utilidades comuns da Fase 0: config, normalização, cache, cliente vLLM, JSONL, logging."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

PKG_DIR = Path(__file__).resolve().parent
REPO = PKG_DIR.parents[3]
PROMPTS = PKG_DIR / "prompts"


# ---------------------------------------------------------------- config / caminhos

def load_config(path: str | os.PathLike | None = None) -> dict:
    p = Path(path) if path else REPO / "configs" / "data" / "phase0.yaml"
    with open(p) as f:
        return yaml.safe_load(f)["phase0"]


def data_dir() -> Path:
    d = Path(os.environ.get("PHASE0_DATA", "data/phase0"))
    d = d if d.is_absolute() else REPO / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_dir() -> Path:
    """Fontes brutas (P0). PHASE0_RAW permite compartilhar entre rodadas (seca/real)."""
    r = os.environ.get("PHASE0_RAW")
    p = (Path(r) if Path(r).is_absolute() else REPO / r) if r else data_dir() / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def log(*a: Any) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- texto

_WS = re.compile(r"\s+")
_TRAIL_PUNCT = re.compile(r"[\s\.\,\;\:\!\?\"'\)\]]+$")


def normalize(s: str) -> str:
    """lower → colapsa espaços → remove pontuação no fim → strip (§P0.2)."""
    s = _WS.sub(" ", (s or "").lower())
    s = _TRAIL_PUNCT.sub("", s)
    return s.strip()


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def extract_json(text: str) -> Any:
    """Primeiro objeto/lista JSON válido no texto (tolera cercas ``` e prosa em volta)."""
    if text is None:
        raise ValueError("texto vazio")
    t = re.sub(r"```(?:json)?", "", text)
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(t[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"sem JSON: {text[:200]!r}")


# ---------------------------------------------------------------- prompts

def load_prompt(cfg: dict, name: str) -> str:
    name = name if name.endswith(".txt") else f"{name}.txt"
    name = name.split("/")[-1]
    return (PROMPTS / name).read_text()


def load_chat_template(cfg: dict, name: str) -> tuple[str, str]:
    """Arquivo no formato 'SYSTEM: ...\\n\\nUSER: ...' → (system, user)."""
    t = load_prompt(cfg, name)
    m = re.match(r"SYSTEM:\s*(.*?)\n\s*\nUSER:\s*(.*)\Z", t, re.S)
    if not m:
        raise ValueError(f"template {name} sem SYSTEM/USER")
    return m.group(1).strip(), m.group(2).rstrip("\n")


def fill(template: str, **values: str) -> str:
    """Substituição literal de placeholders (as chaves são o placeholder exato, ex. '<Q⁺>')."""
    out = template
    for k, v in values.items():
        out = out.replace(k, str(v))
    return out


# ---------------------------------------------------------------- cache

class Cache:
    """Cache em disco por sha256(model ‖ template ‖ input). Escrita atômica (tmp + rename)."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    @staticmethod
    def key(model: str, template_id: str, inputs: Any) -> str:
        blob = json.dumps([model, template_id, inputs], sort_keys=True, ensure_ascii=False)
        return sha256(blob)

    def _path(self, k: str) -> Path:
        return self.root / k[:2] / f"{k}.json"

    def get(self, model: str, template_id: str, inputs: Any) -> Any:
        p = self._path(self.key(model, template_id, inputs))
        if p.exists():
            try:
                return json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def put(self, model: str, template_id: str, inputs: Any, value: Any) -> None:
        p = self._path(self.key(model, template_id, inputs))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False))
        os.replace(tmp, p)

    def call(self, model: str, template_id: str, inputs: Any, fn: Callable[[], Any]) -> Any:
        v = self.get(model, template_id, inputs)
        if v is not None:
            with self._lock:
                self.hits += 1
            return v
        v = fn()
        with self._lock:
            self.misses += 1
        if v is not None:
            self.put(model, template_id, inputs, v)
        return v


def get_cache() -> Cache:
    return Cache(data_dir() / "cache")


# ---------------------------------------------------------------- JSONL

def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> None:
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str))
    os.replace(tmp, p)


# ---------------------------------------------------------------- paralelismo

def pmap(fn: Callable, items: list, workers: int = 32, label: str = "", every: int = 0) -> list:
    """map paralelo com progresso; preserva a ordem."""
    out: list = [None] * len(items)
    every = every or max(1, len(items) // 20)
    done = 0
    lock = threading.Lock()

    def run(i):
        nonlocal done
        out[i] = fn(items[i])
        with lock:
            done += 1
            if label and (done % every == 0 or done == len(items)):
                log(f"{label} {done}/{len(items)}")

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(run, range(len(items))))
    return out


# ---------------------------------------------------------------- cliente vLLM (OpenAI-compatible)

class LLM:
    """Cliente mínimo do servidor vLLM (127.0.0.1). Retry com backoff; contadores de tokens."""

    def __init__(self, url: str, model: str, timeout: float = 600.0):
        import httpx
        self.url = url.rstrip("/")
        self.model = model
        self.http = httpx.Client(timeout=timeout, limits=httpx.Limits(max_connections=256,
                                                                       max_keepalive_connections=256))
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lock = threading.Lock()

    def wait(self, max_s: int = 3600) -> None:
        t0 = time.time()
        while True:
            try:
                if self.http.get(self.url + "/health", timeout=10).status_code == 200:
                    return
            except Exception:
                pass
            if time.time() - t0 > max_s:
                raise RuntimeError(f"{self.url} não respondeu /health")
            time.sleep(5)

    def _post(self, path: str, payload: dict) -> dict:
        delay = 2.0
        for attempt in range(8):
            try:
                r = self.http.post(self.url + path, json=payload)
                if r.status_code == 200:
                    d = r.json()
                    u = d.get("usage") or {}
                    with self._lock:
                        self.stats["calls"] += 1
                        self.stats["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
                        self.stats["completion_tokens"] += u.get("completion_tokens", 0) or 0
                    return d
                if r.status_code == 400:
                    raise ValueError(f"400 de {self.model}: {r.text[:500]}")
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            except ValueError:
                raise
            except Exception as e:  # conexão, timeout
                err = repr(e)
            log(f"vllm {self.model}: {err}; retry {attempt + 1} em {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 120)
        raise RuntimeError(f"{self.model}: falhou após retries")

    def chat(self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 512,
             top_p: float = 1.0, seed: int | None = None, n: int = 1, extra: dict | None = None) -> list[dict]:
        payload = {"model": self.model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "top_p": top_p, "n": n}
        if seed is not None:
            payload["seed"] = seed
        if extra:
            payload.update(extra)
        d = self._post("/v1/chat/completions", payload)
        return [{"text": c["message"].get("content") or "", "finish_reason": c.get("finish_reason")}
                for c in d["choices"]]

    def complete(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 64) -> dict:
        d = self._post("/v1/completions", {"model": self.model, "prompt": prompt,
                                           "temperature": temperature, "max_tokens": max_tokens})
        c = d["choices"][0]
        return {"text": c.get("text") or "", "finish_reason": c.get("finish_reason")}

    def embed(self, texts: list[str]) -> list[list[float]]:
        d = self._post("/v1/embeddings", {"model": self.model, "input": texts})
        return [x["embedding"] for x in sorted(d["data"], key=lambda x: x["index"])]


def no_thinking(model: str) -> dict:
    """Gemma 4 e Qwen3 rodam sem raciocínio (§5.1)."""
    m = model.lower()
    if "gemma-4" in m or "qwen3-8b" in m or m.endswith("qwen3-8b"):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


_THINK = re.compile(r"<think>.*?</think>|<\|channel\|>thought.*?<\|channel\|>|<thought>.*?</thought>", re.S)


def strip_thinking(text: str) -> tuple[str, bool]:
    t2 = _THINK.sub("", text)
    return t2.lstrip(), t2 != text


def url_arg(name: str) -> str:
    """URL de um servidor: --<name>-url ou env PHASE0_<NAME>_URL."""
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument(f"--{name}-url")
    a, _ = ap.parse_known_args()
    v = getattr(a, f"{name.replace('-', '_')}_url") or os.environ.get(f"PHASE0_{name.upper().replace('-', '_')}_URL")
    if not v:
        raise SystemExit(f"falta --{name}-url / PHASE0_{name.upper()}_URL")
    return v
