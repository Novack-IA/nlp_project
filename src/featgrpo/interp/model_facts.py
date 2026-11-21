"""B-U1: confirma os fatos do modelo usados por todas as frentes e grava um JSON.

    PYTHONPATH=src python -m featgrpo.interp.model_facts --out results/frente-b/fatos_modelo.json

Verifica: número de camadas, d_model, caminho das camadas, chat template, tokens da posição
`post` e que `resid_post` da camada L (saída do bloco L) == hidden_states[L+1].
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess

import torch

from featgrpo.interp.common import (
    LAYERS, MODEL_ID, SAE_RELEASE_DIR, get_layers, load_model, sae_id, set_seeds,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/frente-b/fatos_modelo.json")
    args = ap.parse_args()
    set_seeds(0)

    model, tok = load_model()
    layers, layers_path = get_layers(model)
    cfg = model.config.get_text_config()
    n_layers = len(layers)

    # chat template
    msg = [{"role": "user", "content": "How do I kill a Python process?"}]
    rendered = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    ids = tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    toks = tok.convert_ids_to_tokens(ids[0].tolist())
    # sufixo pós-instrução: <end_of_turn>\n<start_of_turn>model\n
    suffix = "<end_of_turn>\n<start_of_turn>model\n"
    suffix_ids = tok(suffix, add_special_tokens=False)["input_ids"]
    suffix_toks = tok.convert_ids_to_tokens(suffix_ids)
    assert ids[0].tolist()[-len(suffix_ids):] == suffix_ids, "sufixo pós-instrução não bate"
    with_sys = tok.apply_chat_template(
        [{"role": "system", "content": "SYS"}] + msg, tokenize=False, add_generation_prompt=True)

    # mapeamento resid_post L == hidden_states[L+1], comparando com hook na saída do bloco
    captured = {}
    hooks = []
    for L in range(n_layers):
        def hook(_m, _inp, out, L=L):
            captured[L] = (out[0] if isinstance(out, tuple) else out).detach()
        hooks.append(layers[L].register_forward_hook(hook))
    with torch.no_grad():
        out = model(input_ids=ids.to(model.device), output_hidden_states=True)
    for h in hooks:
        h.remove()
    hs = out.hidden_states
    # a última posição de hidden_states recebe a norm final no HF; as demais devem ser idênticas
    max_err = max((hs[L + 1] - captured[L]).abs().max().item() for L in range(n_layers - 1))
    exact_selected = {L: torch.equal(hs[L + 1], captured[L]) for L in LAYERS}
    assert all(exact_selected.values()), exact_selected
    last_is_normed = not torch.equal(hs[n_layers], captured[n_layers - 1])

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    import transformers
    facts = {
        "model_id": MODEL_ID,
        "model_class": type(model).__name__,
        "n_layers": n_layers,
        "d_model": cfg.hidden_size,
        "layers_path": layers_path,
        "len_hidden_states": len(hs),
        "resid_post_eq_hidden_states_Lplus1_max_abs_err_L0_to_L32": max_err,
        "resid_post_eq_hidden_states_Lplus1_exact_selected": exact_selected,
        "resid_post_eq_hidden_states_Lplus1_note": (
            "hidden_states[n_layers] tem norm final aplicada" if last_is_normed
            else "hidden_states[n_layers] == saída do último bloco"),
        "layers_selected": LAYERS,
        "layers_frac": {L: round((L + 1) / n_layers, 3) for L in LAYERS},
        "sae_release_dir": SAE_RELEASE_DIR,
        "sae_ids": {L: {w: sae_id(L, w) for w in ("16k", "262k")} for L in LAYERS},
        "chat_template_rendered_example": rendered,
        "chat_template_tokens_example": toks,
        "system_role_rendering": with_sys,
        "post_suffix": suffix,
        "post_suffix_token_ids": suffix_ids,
        "post_suffix_tokens": suffix_toks,
        "bos_added_by_template": toks[0] == tok.bos_token,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "python": platform.python_version()},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "commit": commit,
    }
    with open(args.out, "w") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in facts.items() if "example" not in k}, indent=2,
                     ensure_ascii=False))
    print(rendered)
    print(toks)


if __name__ == "__main__":
    main()
