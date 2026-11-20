"""Sobe o pacote de export_release.py para um dataset PRIVADO no Hugging Face.

Precisa de um token com escrita em HF_WRITE_TOKEN (o HF_TOKEN do .env é só leitura).

    HF_WRITE_TOKEN=... python3 -m featgrpo.data.phase0.upload_release \
        --dir /raid/user_gustavonovack/hf_release/phase0 --repo <org>/<nome>

O repo é criado privado. Para dar acesso ao time: crie-o numa organização da
qual os quatro façam parte, ou depois troque para gated (Settings → Access).
"""
from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--message", default="Fase 0: v5b (final, pares de pedidos, pool) + v5")
    a = ap.parse_args()
    token = os.environ.get("HF_WRITE_TOKEN")
    if not token:
        raise SystemExit("defina HF_WRITE_TOKEN (token com permissão de escrita)")
    api = HfApi(token=token)
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=a.dir, commit_message=a.message)
    print(f"ok: https://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()
