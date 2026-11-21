"""Teste de sanidade do SAE (C-U3 da frente C): convenção de camada, carregamento e posições.

Usa só pedidos escritos à mão e exemplos públicos do Neuronpedia, fora do dataset, então não fere
o pré-registro. Roda numa GPU (LAB150, RTX 4090):
1. Chat template: confere o texto exato e quais tokens são o `fim_pedido` (inst_last) e o
   `pre_resposta` (post_last).
2. Gabarito do Neuronpedia: reproduz ativações publicadas da feature 100 do SAE 16k `medium` da
   camada 22 e confere a convenção "resid_post da camada L = hidden_states[L+1]", comparando com
   hidden_states[L] e hidden_states[L+2].
3. Qualidade do SAE nas duas posições principais, nas 4 camadas: fração da energia não explicada
   (‖x − x̂‖² / ‖x‖²), cosseno entre x e x̂, L0 e ‖x‖.

    python -m featgrpo.interp.sanidade_sae --np-exemplos np_feat100.json --out sanidade_sae.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

LAYERS = (9, 17, 22, 29)
RELEASE = "gemma-scope-2-4b-it-res"
MODEL = "google/gemma-3-4b-it"
PEDIDOS = [  # escritos à mão, benignos; vários com palavras "alarmantes" usadas em sentido inofensivo
    "How do I kill a Python process that is stuck?",
    "How do I terminate a background job in Linux?",
    "What is the best way to shoot a portrait photo at night?",
    "How can I attack the Sicilian Defense as white in chess?",
    "How do I hack together a quick prototype in React?",
    "Can you explain how vaccines train the immune system?",
    "Write a short poem about autumn leaves.",
    "Which household chemicals should never be mixed, for safety reasons?",
    "Where can I find the official address of the city council?",
    "How do I get rid of weeds in my garden without chemicals?",
]


def load_model(token: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, token=token)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", token=token)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda", token=token)
    model.eval()
    return tok, model


def load_sae(layer: int, width: str = "16k"):
    from sae_lens import SAE
    r = SAE.from_pretrained(RELEASE, f"layer_{layer}_width_{width}_l0_medium", device="cuda")
    return r[0] if isinstance(r, tuple) else r


@torch.no_grad()
def hidden(model, ids: torch.Tensor):
    return model(input_ids=ids.to("cuda"), output_hidden_states=True).hidden_states


@torch.no_grad()
def encode(sae, x: torch.Tensor) -> torch.Tensor:
    return sae.encode(x.to(sae.W_enc.dtype)[None])[0]


def template_check(tok, model) -> dict:
    out = {"pedidos": []}
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    for i, p in enumerate(PEDIDOS):
        txt = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        ids = tok(txt, add_special_tokens=False, return_tensors="pt").input_ids
        pieces = tok.convert_ids_to_tokens(ids[0])
        i_eot = int((ids[0] == eot).nonzero()[0])
        item = {"pedido": p, "inst_last": pieces[i_eot - 1], "post_last": pieces[-1], "n_tokens": int(ids.shape[1])}
        if i == 0:
            esperado = f"<bos><start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n"
            out["texto_igual_ao_pre_registro"] = txt == esperado
            out["texto"] = txt
            out["ultimos_6_tokens"] = pieces[-6:]
            out["n_hidden_states"] = len(hidden(model, ids))
        out["pedidos"].append(item)
    out["inst_last_sempre_interrogacao_ou_ponto"] = all(x["inst_last"] in ("?", ".") for x in out["pedidos"])
    return out


def neuronpedia_check(tok, model, sae22, exemplos: dict) -> dict:
    feat, res = int(exemplos["feature"]), []
    for ex in exemplos["exemplos"]:
        toks, vals = ex["tokens"], ex["values"]
        i = int(np.argmax(vals))
        text = "".join(toks[: i + 1])
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
        mesmo = ids.shape[1] == i + 1
        hs = hidden(model, ids)
        medidos = {f"hidden_states[L+{k}]": float(encode(sae22, hs[22 + k][0, -1].float())[feat]) for k in (0, 1, 2)}
        alvo = float(vals[i])
        res.append({"token": toks[i], "posicao": i, "neuronpedia": round(alvo, 2), "retokenizacao_igual": mesmo,
                    "medido": {k: round(v, 2) for k, v in medidos.items()},
                    "erro_relativo_L+1": round(abs(medidos["hidden_states[L+1]"] - alvo) / max(alvo, 1e-6), 4)})
    # L+1 tem de bater em todos os exemplos (erro < 2%), e cada alternativa tem de errar feio (> 5%) em pelo
    # menos um. Exigir que L+1 seja o mais próximo em cada exemplo é frágil: camadas vizinhas às vezes dão
    # valores parecidos para um token por coincidência.
    ok = all(r["erro_relativo_L+1"] < 0.02 for r in res)
    alternativas = {k: max(abs(r["medido"][k] - r["neuronpedia"]) / max(r["neuronpedia"], 1e-6) for r in res)
                    for k in ("hidden_states[L+0]", "hidden_states[L+2]")}
    descartadas = all(v > 0.05 for v in alternativas.values())
    return {"sae": f"{RELEASE}/layer_22_width_16k_l0_medium", "feature": feat, "exemplos": res,
            "maior_erro_L+1": max(r["erro_relativo_L+1"] for r in res),
            "maior_erro_das_alternativas": {k: round(v, 4) for k, v in alternativas.items()},
            "convencao_L+1_confirmada": bool(ok and descartadas)}


@torch.no_grad()
def qualidade(tok, model) -> dict:
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    acts = {"inst_last": [], "post_last": []}
    hs_all = []
    for p in PEDIDOS:
        txt = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        ids = tok(txt, add_special_tokens=False, return_tensors="pt").input_ids
        i_eot = int((ids[0] == eot).nonzero()[0])
        hs_all.append((hidden(model, ids), {"inst_last": i_eot - 1, "post_last": ids.shape[1] - 1}))
    out = {}
    for layer in LAYERS:
        sae = load_sae(layer)
        for pos in acts:
            fvu, cos, l0, norm = [], [], [], []
            for hs, idx in hs_all:
                x = hs[layer + 1][0, idx[pos]].float()
                f = encode(sae, x)
                xh = sae.decode(f[None])[0].float()
                fvu.append(float(((x - xh) ** 2).sum() / (x ** 2).sum()))
                cos.append(float(torch.nn.functional.cosine_similarity(x, xh, dim=0)))
                l0.append(int((f > 0).sum()))
                norm.append(float(x.norm()))
            out[f"L{layer}_{pos}"] = {"fracao_energia_nao_explicada_mediana": round(float(np.median(fvu)), 4),
                                      "cosseno_mediano": round(float(np.median(cos)), 4),
                                      "L0_mediano": int(np.median(l0)), "norma_mediana": round(float(np.median(norm)), 1)}
        del sae
        torch.cuda.empty_cache()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Teste de sanidade do SAE (C-U3).")
    ap.add_argument("--np-exemplos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--env", type=Path, default=Path.home() / "nlp_project" / ".env")
    args = ap.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(args.env)
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "sem GPU"
    import sae_lens
    import transformers

    tok, model = load_model(os.environ.get("HF_TOKEN"))
    res = {"script": "src/featgrpo/interp/sanidade_sae.py", "args": {k: str(v) for k, v in vars(args).items()},
           "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "versoes": {"torch": torch.__version__, "transformers": transformers.__version__,
                       "sae_lens": sae_lens.__version__, "gpu": torch.cuda.get_device_name(0), "host": platform.node()},
           "modelo": {"id": MODEL, "classe": type(model).__name__}}
    res["template"] = template_check(tok, model)
    res["neuronpedia"] = neuronpedia_check(tok, model, load_sae(22), json.loads(args.np_exemplos.read_text()))
    res["qualidade_nas_posicoes_principais"] = qualidade(tok, model)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"template_ok": res["template"]["texto_igual_ao_pre_registro"],
                      "convencao_L+1": res["neuronpedia"]["convencao_L+1_confirmada"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
