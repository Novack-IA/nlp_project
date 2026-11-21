"""B4: qualidade dos SAEs do Gemma Scope 2 nos nossos dados.

    PYTHONPATH=src python -m featgrpo.interp.sae_quality --widths 16k --out results/frente-b/qualidade_saes.csv

Por camada × SAE, só em train + val (o test não entra em decisão de camada):
  - fve:        fração da variância explicada, em todos os tokens dos pedidos (sem o BOS);
  - l0:         média de features ativas por token;
  - dead_frac:  fração de features que nunca ativam nesses tokens;
  - nll_orig / nll_sae / delta_nll: NLL média por token nas respostas **do próprio 4B** (greedy,
    até 64 tokens, a Q⁺ e Q⁰ de train+val, não salvas), com o resid_post da camada L trocado pela
    reconstrução do SAE em todos os tokens menos o BOS;
  - kl_sae: KL(original || com SAE) média por token nessas respostas;
  - delta_nll_zero / kl_zero: o mesmo com a camada zerada (referência de "quão ruim é ruim");
  - frac_recovered: 1 − ΔNLL_sae / ΔNLL_zero.

Com `--positions inst_last post_last` (as duas posições principais do pré-registro da frente C),
grava também `<out>_posicoes.csv`, por camada × SAE × posição:
  - fve_itens: variância explicada **entre itens** no token da posição (1.200 pedidos). É a que
    importa para a frente C: quanto da variação de um pedido para outro o SAE reconstrói;
  - nmse: erro quadrático / ‖x‖², relativo ao vetor inteiro; cos: cosseno médio x × reconstrução;
  - l0: features ativas no token;
  - delta_nll / kl (e *_zero): o splice só naquele token, medido nas respostas do 4B.
  Por que não as respostas r4/r7 do dataset: são do Gemma-4-12B, fora da distribuição do 4B, que
  erra alguns tokens com confiança extrema (até 18 nat); o SAE "amacia" essas previsões e o ΔNLL
  sai negativo (−0,3 a −1,3), o que não mede reconstrução.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import torch

from featgrpo.interp import data
from featgrpo.interp.common import (LAYERS, get_layers, load_model, load_sae, sae_available,
                                    sae_id, set_seeds)
from featgrpo.interp.extract import encode_item, pad_right, run_batch


def _batches(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


@torch.no_grad()
def token_stats(model, tok, layers_mod, encs, L, sae, bs):
    """FVE, L0 e features ativas, somando sobre todos os tokens reais (sem BOS)."""
    sse = 0.0
    n_tok = 0
    l0_sum = 0.0
    ever = torch.zeros(sae.d_sae, dtype=torch.bool, device="cuda")
    s1 = None
    s2 = 0.0
    for b in _batches(encs, bs):
        cap = run_batch(model, layers_mod, [L], b, tok.pad_token_id)
        xs = torch.cat([cap[L][j, 1:len(e.input_ids)] for j, e in enumerate(b)]).float()
        f = sae.encode(xs)
        rec = sae.decode(f)
        sse += ((xs - rec) ** 2).sum().item()
        s1 = xs.sum(0) if s1 is None else s1 + xs.sum(0)
        s2 += (xs ** 2).sum().item()
        n_tok += xs.shape[0]
        l0_sum += (f > 0).sum().item()
        ever |= (f > 0).any(0)
    sst = s2 - (s1 ** 2).sum().item() / n_tok
    return {"fve": 1 - sse / sst, "l0": l0_sum / n_tok,
            "dead_frac": 1 - ever.float().mean().item(), "n_tokens": n_tok}


@torch.no_grad()
def position_stats(model, tok, layers_mod, encs, L, sae, pos, bs):
    """Reconstrução do SAE só no token da posição `pos` (uma por item)."""
    xs = []
    for b in _batches(encs, bs):
        cap = run_batch(model, layers_mod, [L], b, tok.pad_token_id)
        xs += [cap[L][j, e.positions[pos][0]] for j, e in enumerate(b) if e.positions.get(pos)]
    x = torch.stack(xs).float()
    f = sae.encode(x)
    rec = sae.decode(f)
    sse = ((x - rec) ** 2).sum().item()
    return {"fve_itens": 1 - sse / ((x - x.mean(0)) ** 2).sum().item(),
            "nmse": sse / (x ** 2).sum().item(),
            "cos": torch.nn.functional.cosine_similarity(x, rec, dim=-1).mean().item(),
            "l0": (f > 0).sum(1).float().mean().item(), "n_itens": x.shape[0]}


@torch.no_grad()
def generate_own(model, tok, prompts: list[str], max_new: int, bs: int):
    """Respostas greedy do próprio Gemma-3-4B, como `Encoded` (ids do pedido + ids gerados).

    Ficam só em memória: servem para medir ΔNLL/KL na distribuição do próprio modelo."""
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")}
    encs = [encode_item(tok, {"text": p}) for p in prompts]
    out = []
    for b in _batches(encs, bs):
        T = max(len(e.input_ids) for e in b)
        ids = torch.full((len(b), T), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, e in enumerate(b):
            ids[i, T - len(e.input_ids):] = torch.tensor(e.input_ids)
            att[i, T - len(e.input_ids):] = 1
        g = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=max_new,
                           do_sample=False)[:, T:].tolist()
        for e, gen in zip(b, g):
            k = next((j for j, t in enumerate(gen) if t in stop), len(gen))
            if k == 0:
                continue
            n = len(e.input_ids)
            e.input_ids = e.input_ids + gen[:k]
            e.positions["resp_mean"] = list(range(n, n + k))
            out.append(e)
    return out


def _splice_hook(b, mode, sae, pos=None):
    """Troca a saída do bloco pela reconstrução (mode="sae") ou por zero, em todos os tokens menos
    o BOS (pos=None) ou só no token da posição `pos` de cada item. Padding à direita."""
    def hook(_m, _i, out):
        x = out[0] if isinstance(out, tuple) else out
        new = sae(x.float()).to(x.dtype) if mode == "sae" else torch.zeros_like(x)
        keep = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        for i, e in enumerate(b):
            if pos is None:
                keep[i, 1:] = False      # o BOS fica com o residual original
            else:
                keep[i, e.positions[pos]] = False
        new = torch.where(keep[..., None], x, new)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
    return hook


@torch.no_grad()
def splice_metrics(model, tok, layers_mod, encs, L, mode, sae=None, bs=8, pos=None):
    """Nos tokens de resposta: NLL original, NLL com a camada L trocada (mode: sae | zero; em todos
    os tokens ou só na posição `pos`) e KL(original || trocada) média por token."""
    nll_o = nll_m = kl = 0.0
    n = 0
    for b in _batches(encs, bs):
        ids, att = pad_right(b, tok.pad_token_id)
        ids, att = ids.cuda(), att.cuda()
        lo = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        h = layers_mod[L].register_forward_hook(_splice_hook(b, mode, sae, pos))
        try:
            lm = model(input_ids=ids, attention_mask=att, use_cache=False).logits
        finally:
            h.remove()
        for i, e in enumerate(b):
            r = [t for t in e.positions["resp_mean"] if t > 0]
            prev = [t - 1 for t in r]
            tgt = ids[i, r][:, None]
            po = torch.log_softmax(lo[i, prev].float(), -1)
            pm = torch.log_softmax(lm[i, prev].float(), -1)
            nll_o += -po.gather(-1, tgt).sum().item()
            nll_m += -pm.gather(-1, tgt).sum().item()
            kl += (po.exp() * (po - pm)).sum().item()
            n += len(r)
        del lo, lm
    return nll_o / n, nll_m / n, kl / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    ap.add_argument("--widths", nargs="+", default=["16k"])
    ap.add_argument("--n-pairs", type=int, default=600, help="pares de train+val para FVE/L0")
    ap.add_argument("--n-gen", type=int, default=150, help="pares de train+val cujos Q⁺ e Q⁰ o 4B responde (ΔNLL/KL)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64, help="tamanho das respostas geradas")
    ap.add_argument("--out", default="results/frente-b/qualidade_saes.csv")
    ap.add_argument("--positions", nargs="*", default=[],
                    help="posições de um token para medir à parte (ex.: inst_last post_last)")
    ap.add_argument("--skip-full", action="store_true",
                    help="pula as medidas em todos os tokens (só as das --positions)")
    args = ap.parse_args()
    set_seeds(0)
    import random

    model, tok = load_model()
    layers_mod, _ = get_layers(model)

    pairs = data.load_pairs("train") + data.load_pairs("val")
    random.Random(0).shuffle(pairs)
    p_encs = [encode_item(tok, it) for it in data.prompt_items(pairs[:args.n_pairs])]
    gen_prompts = [it["text"] for it in data.prompt_items(pairs[args.n_pairs:args.n_pairs + args.n_gen])]
    r_encs = generate_own(model, tok, gen_prompts, args.max_new_tokens, args.batch_size)
    n_resp_tok = sum(len(e.positions["resp_mean"]) for e in r_encs)
    print(f"{len(p_encs)} pedidos; {len(r_encs)} respostas do 4B ({n_resp_tok} tokens)", flush=True)

    rows, pos_rows = [], []
    for L in args.layers:
        if not args.skip_full:
            nll_orig, nll_zero, kl_zero = splice_metrics(model, tok, layers_mod, r_encs, L, "zero")
        zero_pos = {p: splice_metrics(model, tok, layers_mod, r_encs, L, "zero", pos=p)
                    for p in args.positions}
        for w in args.widths:
            if not sae_available(L, w):
                continue
            t0 = time.time()
            sae = load_sae(L, w)
            for p in args.positions:
                ps = position_stats(model, tok, layers_mod, p_encs, L, sae, p, args.batch_size)
                no, ns, ks = splice_metrics(model, tok, layers_mod, r_encs, L, "sae", sae, pos=p)
                nz, kz = zero_pos[p][1], zero_pos[p][2]
                prow = {"layer": L, "sae": sae_id(L, w), "width": w, "position": p, **ps,
                        "nll_orig": no, "delta_nll": ns - no, "kl_sae": ks,
                        "delta_nll_zero": nz - no, "kl_zero": kz}
                print({k: round(v, 4) if isinstance(v, float) else v for k, v in prow.items()},
                      flush=True)
                pos_rows.append(prow)
            if args.skip_full:
                del sae
                torch.cuda.empty_cache()
                continue
            st = token_stats(model, tok, layers_mod, p_encs, L, sae, args.batch_size)
            _, nll_sae, kl_sae = splice_metrics(model, tok, layers_mod, r_encs, L, "sae", sae)
            row = {"layer": L, "sae": sae_id(L, w), "width": w, **st,
                   "nll_orig": nll_orig, "nll_sae": nll_sae, "delta_nll": nll_sae - nll_orig,
                   "kl_sae": kl_sae, "delta_nll_zero": nll_zero - nll_orig, "kl_zero": kl_zero,
                   "frac_recovered": 1 - (nll_sae - nll_orig) / (nll_zero - nll_orig)}
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()},
                  f"{time.time() - t0:.0f}s", flush=True)
            rows.append(row)
            del sae
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for path, rr in ((out, rows), (out.with_name(out.stem + "_posicoes.csv"), pos_rows)):
        if not rr:
            continue
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rr[0]))
            if write_header:
                wr.writeheader()
            wr.writerows(rr)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    meta = out.with_suffix(".json")
    prev = json.loads(meta.read_text()) if meta.exists() else []
    prev.append({"args": vars(args), "commit": commit, "n_prompts": len(p_encs),
                 "n_responses": len(r_encs), "n_response_tokens": n_resp_tok, "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
    meta.write_text(json.dumps(prev, indent=2))
    print("ok:", out)


if __name__ == "__main__":
    main()
