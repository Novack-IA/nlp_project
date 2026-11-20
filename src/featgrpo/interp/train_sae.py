"""SAE próprio (A5): BatchTopK no resid_post de uma camada do Gemma-3-4B-IT, ativações geradas na hora.

    python -m featgrpo.interp.train_sae train --layer 22 --out DIR --corpus DIR [--tokens 2e8]
    python -m featgrpo.interp.train_sae eval  --layer 22 --out DIR --corpus DIR [--ref-gemmascope]

Convenções:
  - resid_post da camada L = saída do bloco L (= hidden_states[L+1] do HF).
  - O token BOS (posição 0) fica de fora (norma anômala).
  - Treino em espaço normalizado (E||x||² = d); o fator é dobrado nos pesos salvos, então o SAE salvo
    recebe a ativação crua.
  - Saída no formato do SAELens (cfg.json + sae_weights.safetensors, arquitetura jumprelu com o limiar
    global do BatchTopK). Retomável: checkpoint a cada --ckpt-every tokens.
Rodar com o venv do vLLM (torch + transformers).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL = "google/gemma-3-4b-it"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- modelo e captura

class _Stop(Exception):
    pass


def load_model(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.to(device).eval()
    cfg = model.config
    n = getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers")
    layers = None
    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and len(mod) == n \
                and "vision" not in name:
            layers = mod
            log(f"camadas: {name} ({n})")
            break
    assert layers is not None, "não achei a ModuleList de camadas do modelo de texto"
    return tok, model, layers


class Capture:
    """Hook na saída do bloco L; opcionalmente interrompe o forward (stop) ou substitui (replace_fn)."""

    def __init__(self, layer_mod, stop=True):
        self.out = None
        self.stop = stop
        self.replace_fn = None
        self.h = layer_mod.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if self.replace_fn is not None:
            x2 = self.replace_fn(x)
            return (x2,) + tuple(out[1:]) if isinstance(out, tuple) else x2
        self.out = x
        if self.stop:
            raise _Stop()


def conv_tokens(tok, messages, ctx):
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)[:ctx]


def batches_from_corpus(tok, path, ctx, bs, start_line=0):
    """Gera (input_ids, attention_mask, linha) em lotes de bs conversas."""
    buf = []
    with open(path) as f:
        for i, l in enumerate(f):
            if i < start_line:
                continue
            ids = conv_tokens(tok, json.loads(l)["messages"], ctx)
            if len(ids) < 4:
                continue
            buf.append(ids)
            if len(buf) == bs:
                yield pad(buf), i + 1
                buf = []
    if buf:
        yield pad(buf), None


def pad(seqs):
    L = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    m = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        m[i, :len(s)] = 1
    return ids, m


@torch.no_grad()
def acts(model, cap, ids, mask, device):
    try:
        model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
    except _Stop:
        pass
    x = cap.out
    keep = mask.to(device).bool().clone()
    keep[:, 0] = False                       # sem BOS
    return x[keep].float()


# ---------------------------------------------------------------- SAE

class BatchTopK(torch.nn.Module):
    def __init__(self, d, m, k):
        super().__init__()
        self.d, self.m, self.k = d, m, k
        W = torch.randn(d, m) / math.sqrt(d)
        self.W_enc = torch.nn.Parameter(W.clone())
        self.W_dec = torch.nn.Parameter(F.normalize(W.T.clone(), dim=1))
        self.b_enc = torch.nn.Parameter(torch.zeros(m))
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.register_buffer("threshold", torch.tensor(0.0))

    def pre(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def encode_topk(self, x):
        p = self.pre(x)
        B = x.shape[0]
        v, idx = p.flatten().topk(self.k * B)
        f = torch.zeros_like(p.flatten()).scatter_(0, idx, v).view_as(p)
        return f, p, v

    def encode_jump(self, x):
        p = self.pre(x)
        return p * (p > self.threshold)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    @torch.no_grad()
    def renorm(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


def save_saelens(sae: BatchTopK, scale: float, layer: int, out: Path, extra: dict):
    from safetensors.torch import save_file
    out.mkdir(parents=True, exist_ok=True)
    s = scale
    # pré-ativação em espaço normalizado: (x*s - b) W + c  =  (x - b/s) (sW) + c ; decode: (f Wd + b)/s
    W_enc = (sae.W_enc.data * s).contiguous()
    W_dec = (sae.W_dec.data / s).contiguous()
    b_dec = (sae.b_dec.data / s).contiguous()
    th = torch.full((sae.m,), float(sae.threshold))
    save_file({"W_enc": W_enc.cpu(), "W_dec": W_dec.cpu(), "b_enc": sae.b_enc.data.cpu().contiguous(),
               "b_dec": b_dec.cpu(), "threshold": th}, str(out / "sae_weights.safetensors"))
    cfg = {"architecture": "jumprelu", "d_in": sae.d, "d_sae": sae.m, "dtype": "float32", "device": "cpu",
           "model_name": MODEL, "hook_name": f"blocks.{layer}.hook_resid_post", "hook_layer": layer,
           "hf_hook_name": f"language_model.layers.{layer} (saída do bloco = hidden_states[{layer + 1}])",
           "apply_b_dec_to_input": True, "normalize_activations": "none", "context_size": extra.get("ctx"),
           "prepend_bos": True, "training": {"type": "batchtopk", "k": sae.k, "scale_folded": s, **extra}}
    (out / "cfg.json").write_text(json.dumps(cfg, indent=1))


# ---------------------------------------------------------------- treino

def train(a):
    dev = torch.device("cuda")
    torch.manual_seed(0); random.seed(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok, model, layers = load_model(dev)
    cap = Capture(layers[a.layer], stop=True)
    d = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
    sae = BatchTopK(d, a.width, a.k).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=a.lr, betas=(0.9, 0.999))
    total_steps = int(a.tokens // a.batch)
    st = {"tokens": 0, "step": 0, "line": 0, "scale": None}
    ck = out / "ckpt.pt"
    if ck.exists():
        c = torch.load(ck, map_location=dev)
        sae.load_state_dict(c["sae"]); opt.load_state_dict(c["opt"]); st = c["st"]
        log(f"retomando: {st}")
    last_fire = torch.zeros(a.width, device=dev, dtype=torch.long) + st["tokens"]
    corpus = Path(a.corpus) / "train.jsonl"
    gen = batches_from_corpus(tok, corpus, a.ctx, a.seqs, start_line=st["line"])
    buf = torch.empty(0, d, device=dev)
    t0, tok0 = time.time(), st["tokens"]
    next_ck = st["tokens"] + a.ckpt_every
    k_aux, dead_after = 512, 5_000_000

    def lr_at(step):
        w = min(1.0, (step + 1) / 1000)
        dec_start = int(total_steps * 0.8)
        dec = 1.0 if step < dec_start else max(0.0, (total_steps - step) / max(1, total_steps - dec_start))
        return a.lr * w * dec

    exhausted = False
    while st["tokens"] < a.tokens and not exhausted:
        # enche o buffer
        while buf.shape[0] < a.buffer:
            try:
                (ids, m), line = next(gen)
            except StopIteration:
                exhausted = True
                break
            x = acts(model, cap, ids, m, dev)
            buf = torch.cat([buf, x])
            if line is not None:
                st["line"] = line
        if st["scale"] is None:
            st["scale"] = float(math.sqrt(d / buf.pow(2).sum(1).mean().item()))
            with torch.no_grad():
                sae.b_dec.data = (buf[:200_000] * st["scale"]).median(0).values
            log(f"escala de normalização: {st['scale']:.5f}")
        buf = buf[torch.randperm(buf.shape[0], device=dev)]
        use = buf.shape[0] if exhausted else buf.shape[0] // 2
        for i in range(0, use - a.batch + 1, a.batch):
            x = buf[i:i + a.batch] * st["scale"]
            f, p, v = sae.encode_topk(x)
            xh = sae.decode(f)
            e = x - xh
            mse = e.pow(2).sum(1).mean()
            fvu = mse / (x - x.mean(0)).pow(2).sum(1).mean()
            fired = (f > 0).any(0)
            last_fire[fired] = st["tokens"]
            dead = (st["tokens"] - last_fire) > dead_after
            aux = torch.tensor(0.0, device=dev)
            if dead.sum() > 0:
                pd = p[:, dead]
                kk = min(k_aux, pd.shape[1])
                av, ai = pd.topk(kk, dim=1)
                fa = torch.zeros_like(pd).scatter_(1, ai, av)
                eh = fa @ sae.W_dec[dead]
                aux = (e.detach() - eh).pow(2).sum(1).mean() / e.detach().pow(2).sum(1).mean().clamp_min(1e-6)
            loss = fvu + aux / 32
            for g in opt.param_groups:
                g["lr"] = lr_at(st["step"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # remove a componente do gradiente paralela às linhas do decoder
            with torch.no_grad():
                W, G = sae.W_dec.data, sae.W_dec.grad
                G -= (G * W).sum(1, keepdim=True) * W
            opt.step()
            sae.renorm()
            with torch.no_grad():
                mn = v.min()
                sae.threshold.mul_(0.999).add_(0.001 * mn) if st["step"] > 0 else sae.threshold.fill_(mn)
            st["tokens"] += a.batch
            st["step"] += 1
            if st["step"] % 500 == 0:
                rate = (st["tokens"] - tok0) / (time.time() - t0)
                log(f"L{a.layer} step {st['step']}/{total_steps} tok {st['tokens'] / 1e6:.1f}M fvu {fvu.item():.4f} "
                    f"aux {aux.item():.3f} dead {dead.float().mean().item():.3%} thr {float(sae.threshold):.4f} "
                    f"lr {lr_at(st['step']):.2e} {rate / 1e3:.1f}k tok/s")
            if st["tokens"] >= next_ck:
                torch.save({"sae": sae.state_dict(), "opt": opt.state_dict(), "st": st}, ck)
                next_ck += a.ckpt_every
            if st["tokens"] >= a.tokens:
                break
        buf = buf[use:]
    torch.save({"sae": sae.state_dict(), "opt": opt.state_dict(), "st": st}, ck)
    save_saelens(sae, st["scale"], a.layer, out / "sae", {"ctx": a.ctx, "tokens": st["tokens"], "width": a.width,
                                                          "lr": a.lr, "batch": a.batch, "corpus": str(a.corpus)})
    (out / "DONE").write_text(json.dumps(st))
    log(f"fim do treino: {st}")


# ---------------------------------------------------------------- avaliação

def load_jumprelu(path: Path, dev):
    """SAE no formato SAELens (nosso) ou Gemma Scope 2 (params.safetensors); devolve encode/decode em x cru."""
    from safetensors.torch import load_file
    f = path / "sae_weights.safetensors"
    if not f.exists():
        f = path / "params.safetensors"
    w = {k.lower(): v.to(dev).float() for k, v in load_file(str(f)).items()}
    cfgp = path / "cfg.json" if (path / "cfg.json").exists() else path / "config.json"
    cfg = json.loads(cfgp.read_text()) if cfgp.exists() else {}
    W_enc, W_dec, b_enc, b_dec, th = w["w_enc"], w["w_dec"], w["b_enc"], w["b_dec"], w["threshold"]
    sub = bool(cfg.get("apply_b_dec_to_input", False))

    def enc(x):
        p = (x - b_dec if sub else x) @ W_enc + b_enc
        return p * (p > th)

    def dec(f):
        return f @ W_dec + b_dec
    return enc, dec, W_enc.shape[1], cfg


@torch.no_grad()
def evaluate(a):
    dev = torch.device("cuda")
    out = Path(a.out)
    tok, model, layers = load_model(dev)
    saes = {"proprio": out / "sae"}
    if a.ref_gemmascope:
        from huggingface_hub import snapshot_download
        for name in a.ref_gemmascope.split(","):
            p = snapshot_download("google/gemma-scope-2-4b-it", allow_patterns=[f"{name}/*"])
            saes[name] = Path(p) / name
    res = {"layer": a.layer}
    texts = [json.loads(l)["messages"] for l in open(Path(a.corpus) / "eval.jsonl")][: a.eval_convs]
    pairs = []
    pp = Path(__file__).resolve().parents[3] / "datasets" / "phase0" / "v5b" / "pairs_prompt_level.jsonl"
    for l in open(pp):
        p = json.loads(l)
        if p["split"] in ("train", "val"):
            pairs += [[{"role": "user", "content": p["q_harm"]}], [{"role": "user", "content": p["q_border"]}]]
    pairs = pairs[: a.eval_convs]
    for sname, spath in saes.items():
        enc, dec, m, cfg = load_jumprelu(spath, dev)
        r = {"d_sae": m}
        for dname, convs in (("ultrachat_eval", texts), ("nossos_pedidos", pairs)):
            fired = torch.zeros(m, dtype=torch.bool, device=dev)
            se = tot = l0 = n = 0.0
            nll_c = nll_s = ntok = 0.0
            cap = Capture(layers[a.layer], stop=False)
            for i in range(0, len(convs), 8):
                seqs = [conv_tokens(tok, c, a.ctx) for c in convs[i:i + 8]]
                ids, msk = pad(seqs)
                ids, msk = ids.to(dev), msk.to(dev)
                cap.replace_fn = None
                lg = model(input_ids=ids, attention_mask=msk, use_cache=False).logits.float()
                x = cap.out
                keep = msk.bool().clone(); keep[:, 0] = False
                xs = x[keep].float()
                f = enc(xs); xh = dec(f)
                se += (xs - xh).pow(2).sum().item(); tot += (xs - xs.mean(0)).pow(2).sum().item()
                l0 += (f > 0).sum().item(); n += xs.shape[0]; fired |= (f > 0).any(0)

                def rep(h):
                    hk = h.clone()
                    hf = h.float()
                    rr = dec(enc(hf)).to(h.dtype)
                    kk = keep.unsqueeze(-1)
                    return torch.where(kk, rr, hk)
                cap.replace_fn = rep
                lg2 = model(input_ids=ids, attention_mask=msk, use_cache=False).logits.float()
                cap.replace_fn = None
                tgt = ids[:, 1:]; mk = msk[:, 1:].bool()
                lc = F.cross_entropy(lg[:, :-1].transpose(1, 2), tgt, reduction="none")[mk]
                ls = F.cross_entropy(lg2[:, :-1].transpose(1, 2), tgt, reduction="none")[mk]
                nll_c += lc.sum().item(); nll_s += ls.sum().item(); ntok += mk.sum().item()
            cap.h.remove()
            r[dname] = {"fvu": se / max(tot, 1e-9), "var_explained": 1 - se / max(tot, 1e-9), "l0": l0 / max(n, 1),
                        "frac_dead_in_set": 1 - fired.float().mean().item(), "nll_clean": nll_c / ntok,
                        "nll_spliced": nll_s / ntok, "delta_nll": (nll_s - nll_c) / ntok, "tokens": int(n)}
            log(f"{sname} {dname}: {r[dname]}")
        res[sname] = r
    (out / "eval.json").write_text(json.dumps(res, indent=1))
    log("eval ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--width", type=int, default=16384)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--tokens", type=float, default=2e8)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seqs", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--buffer", type=int, default=1_000_000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ckpt-every", type=float, default=2e7)
    ap.add_argument("--ref-gemmascope", default="")
    ap.add_argument("--eval-convs", type=int, default=400)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
