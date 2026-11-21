"""Constantes e utilitários compartilhados da frente B (modelo, camadas, SAEs)."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "google/gemma-3-4b-it"
SAE_REPO = "google/gemma-scope-2-4b-it"
SAE_RELEASE_DIR = "resid_post"           # pasta no repo do HF; SAELens: gemma-scope-2-4b-it-resid_post
SAE_L0 = "medium"                         # L0≈60; existe em 16k e 262k nas 4 camadas
# 34 camadas; as únicas com todas as larguras (16k/65k/262k/1m × small/medium/big) no HF.
LAYERS = [9, 17, 22, 29]                  # ≈ 25 / 50 / 65 / 85 % de profundidade

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = Path(os.environ.get("ARTIFACTS", REPO / "artifacts"))
ACTS = ARTIFACTS / "acts" / "gemma-3-4b-it"


def set_seeds(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sae_id(layer: int, width: str, l0: str = SAE_L0) -> str:
    return f"layer_{layer}_width_{width}_l0_{l0}"


def load_model(dtype=torch.bfloat16, device="cuda"):
    """Carrega o Gemma-3-4B-IT (checkpoint multimodal; usamos só o texto)."""
    from dotenv import load_dotenv
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    load_dotenv(REPO / ".env")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=device, attn_implementation="eager")
    model.eval()
    model.requires_grad_(False)
    return model, tok


def get_layers(model):
    """Devolve (ModuleList de blocos do LM, caminho em texto)."""
    for path in ("model.language_model.layers", "language_model.model.layers",
                 "model.language_model.model.layers", "language_model.layers", "model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList):
            return obj, path
    raise RuntimeError("não achei as camadas do LM")


class JumpReLUSAE(torch.nn.Module):
    """SAE JumpReLU do Gemma Scope 2, carregado direto do params.safetensors."""

    def __init__(self, w_enc, b_enc, w_dec, b_dec, threshold):
        super().__init__()
        self.w_enc = torch.nn.Parameter(w_enc, requires_grad=False)
        self.b_enc = torch.nn.Parameter(b_enc, requires_grad=False)
        self.w_dec = torch.nn.Parameter(w_dec, requires_grad=False)
        self.b_dec = torch.nn.Parameter(b_dec, requires_grad=False)
        self.threshold = torch.nn.Parameter(threshold, requires_grad=False)

    @property
    def d_sae(self) -> int:
        return self.w_enc.shape[1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x.to(self.w_enc.dtype) @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.w_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.encode(x))


def load_sae(layer: int, width: str, l0: str = SAE_L0, device="cuda",
             dtype=torch.float32) -> JumpReLUSAE:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(SAE_REPO, f"{SAE_RELEASE_DIR}/{sae_id(layer, width, l0)}/params.safetensors")
    p = {k.lower(): v for k, v in load_file(path).items()}
    sae = JumpReLUSAE(p["w_enc"], p["b_enc"], p["w_dec"], p["b_dec"], p["threshold"])
    return sae.to(device=device, dtype=dtype)
