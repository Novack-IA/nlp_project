"""Baixa os pesos para HF_HUB_CACHE, na ordem dada, marcando cada modelo pronto com .ok_<org>--<nome>.

    python -m featgrpo.data.phase0.download_models MODEL [MODEL ...]

Só safetensors quando existem (ignora original/, consolidated*, .bin/.pth duplicados).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download


def marker(model: str) -> Path:
    return Path(os.environ["HF_HUB_CACHE"]) / f".ok_{model.replace('/', '--')}"


def main():
    tok = os.environ.get("HF_TOKEN")
    for m in sys.argv[1:]:
        if marker(m).exists():
            print(f"[download] {m} já pronto", flush=True)
            continue
        t0 = time.time()
        files = list_repo_files(m, token=tok)
        ignore = ["original/*", "consolidated*", "*.gguf", "onnx/*", "*.onnx", "*.msgpack", "*.h5", "flax_model*", "tf_model*"]
        if any(f.endswith(".safetensors") for f in files):
            ignore += ["*.bin", "*.pth", "*.pt"]
        for attempt in range(5):
            try:
                snapshot_download(m, token=tok, ignore_patterns=ignore, max_workers=16)
                break
            except Exception as e:  # rede
                print(f"[download] {m}: {e!r}; tentativa {attempt + 1}", flush=True)
                time.sleep(30)
        else:
            sys.exit(f"[download] {m}: falhou")
        marker(m).touch()
        print(f"[download] {m} ok em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
