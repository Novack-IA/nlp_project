"""B2: extrator único de ativações (resid_post) do Gemma-3-4B-IT + latentes dos SAEs.

    PYTHONPATH=src python -m featgrpo.interp.extract --set pairs            # 2.428 × 2 pedidos
    PYTHONPATH=src python -m featgrpo.interp.extract --set pairs --limit 64 --out-root /tmp/acts
    PYTHONPATH=src python -m featgrpo.interp.extract --set cells2x2 --widths 16k 262k
    PYTHONPATH=src python -m featgrpo.interp.extract --set controls

Saída (contrato da seção 4 do AGENTS.md):
    {root}/resid_post/L{L:02d}/{conjunto}__{posicao}.npy       float32 [N, d_model]
    {root}/index/{conjunto}.jsonl                               linha i ↔ linha i do .npy
    {root}/resid_post/L{L:02d}/{conjunto}__{posicao}__{sae}.npz scipy CSR float32 [N, d_sae]
        sae = 16k | 262k (Gemma Scope 2, l0_medium) | proprio  (convenção de interp/cache_io.py)

Posições de pedido (principais no pré-registro da frente C: inst_last e post_last):
    inst_last  último token do texto do usuário, logo antes de <end_of_turn> (`inst_last_token_id` no
               índice, para conferir que é o mesmo token nas 4 células do 2×2)
    post       média dos tokens do sufixo "<end_of_turn>\\n<start_of_turn>model\\n"
    post_last  último token desse sufixo
    mean       média dos tokens do conteúdo do pedido (sem template)
    trig       média dos tokens que cobrem o gatilho (NaN se não alinhou; ver trigger_aligned).
               Pares originais e controles.
    slot_forma / slot_intencao   média dos tokens das lacunas do 2×2, pelos `spans` do item
               (posições em caracteres no `text`). No cells_2x2.jsonl da frente A elas vêm em
               `cells[c].slots.{form,intent}`; no formato de render_cells, em `cells[c].spans.{forma,intencao}`.
               Substituem `trig` no 2×2.
Lotes: padding à DIREITA, com as células de um mesmo grupo sempre no mesmo lote. Assim os tokens do
    prefixo comum às células (tudo antes da primeira diferença) saem bit a bit iguais, e um efeito de
    intenção em `slot_forma` (quando a lacuna de forma vem antes da de intenção) dá exatamente 0.
Posições de resposta (itens com "response", teacher forcing):
    resp_mean  média dos tokens da resposta;  resp_k8  média dos 8 primeiros
Latentes: `{pos}` = SAE aplicado ao residual médio da posição (o mesmo vetor do .npy);
    `{pos}_tokmean` / `{pos}_tokmax` = SAE aplicado a cada token da posição, depois média / máximo.
    A média de muitos tokens sai da distribuição do SAE (L0 de `mean` ≈ 6–9 contra ≈ 50 por token),
    então para estatística por feature prefira `_tokmean`/`_tokmax`. A escolha é da frente C.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from featgrpo.interp.common import ACTS, LAYERS, MODEL_ID, SAE_REPO, get_layers, sae_id, set_seeds

POST_SUFFIX = "<end_of_turn>\n<start_of_turn>model\n"
BASE_POSITIONS = ("inst_last", "post", "post_last", "mean")
PROMPT_POSITIONS = BASE_POSITIONS + ("trig",)
SLOT_POSITIONS = ("slot_forma", "slot_intencao")
RESP_POSITIONS = ("resp_mean", "resp_k8")
# posições de vários tokens que também ganham latentes token a token (média e máximo)
TOKEN_POOLED = ("post", "mean", "trig", "slot_forma", "slot_intencao", "resp_mean", "resp_k8")

# ---------------------------------------------------------------- alinhamento do gatilho

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_stemmer = None


def _stem(w: str) -> str:
    global _stemmer
    if _stemmer is None:
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
    return _stemmer.stem(w.lower())


def locate_trigger(text: str, trigger: str | None) -> tuple[list[tuple[int, int]], str]:
    """Spans de caractere do gatilho em `text` e o método usado.

    1. exact: busca sem diferenciar maiúsculas (primeira ocorrência);
    2. stem: janela contígua de palavras cujos radicais Porter batem com os do gatilho;
    3. stem_partial: palavras do texto cujo radical está entre os radicais do gatilho;
    4. none: sem alinhamento (registrado no índice, não descartado).
    """
    if not trigger or not trigger.strip():
        return [], "none"
    trig = trigger.strip()
    # limite de palavra: "kill" não casa dentro de "skill" nem de "killing" (esse vai por radical)
    m = re.search(r"(?<![A-Za-z0-9])" + re.escape(trig) + r"(?![A-Za-z0-9])", text, re.IGNORECASE)
    if m:
        return [(m.start(), m.end())], "exact"
    words = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(text)]
    tstems = [_stem(w) for w in _WORD.findall(trig)]
    if not tstems:
        return [], "none"
    stems = [_stem(w) for w, _, _ in words]
    n = len(tstems)
    for k in range(len(words) - n + 1):
        if stems[k:k + n] == tstems:
            return [(words[k][1], words[k + n - 1][2])], "stem"
    tset = set(tstems)
    spans = [(s, e) for (w, s, e), st in zip(words, stems) if st in tset]
    if spans:
        return spans, "stem_partial"
    return [], "none"


# ---------------------------------------------------------------- formatação e posições

@dataclass
class Encoded:
    input_ids: list[int]
    positions: dict[str, list[int]] = field(default_factory=dict)   # nome -> índices de token
    meta: dict = field(default_factory=dict)


def _tokens_in_spans(offsets, spans) -> list[int]:
    out = []
    for t, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        if any(a < e and b > s for s, e in spans):
            out.append(t)
    return out


def encode_item(tok, item: dict, trigger_key: str = "trigger") -> Encoded:
    """Aplica o chat template e calcula os índices de token de cada posição."""
    text = item["text"].strip()      # o template do Gemma 3 faz trim do conteúdo
    msgs = [{"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert prompt.endswith(POST_SUFFIX), prompt[-60:]
    rendered = prompt
    resp = item.get("response")
    resp_start = None
    if resp is not None:
        resp_start = len(rendered)
        rendered = rendered + resp.strip()
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]

    c0 = prompt.rfind(text, 0, len(prompt) - len(POST_SUFFIX))
    assert c0 >= 0, "conteúdo não encontrado no prompt renderizado"
    c1 = c0 + len(text)
    s0 = len(prompt) - len(POST_SUFFIX)
    pos: dict[str, list[int]] = {}
    pos["mean"] = _tokens_in_spans(offs, [(c0, c1)])
    pos["inst_last"] = pos["mean"][-1:]
    post = [t for t, (a, b) in enumerate(offs) if a >= s0 and b <= len(prompt) and b > a]
    pos["post"] = post
    pos["post_last"] = post[-1:]
    spans, method = locate_trigger(text, item.get(trigger_key))
    pos["trig"] = _tokens_in_spans(offs, [(c0 + s, c0 + e) for s, e in spans])
    if item.get("spans"):                  # 2×2: posições das lacunas, relativas ao `text` original
        lead = len(item["text"]) - len(item["text"].lstrip())
        for k in ("forma", "intencao"):
            s, e = item["spans"][k]
            assert item["text"][s:e].strip(), (item.get("item_id"), k, "span vazio")
            pos[f"slot_{k}"] = _tokens_in_spans(offs, [(c0 + s - lead, c0 + e - lead)])
    if resp is not None:
        r = [t for t, (a, b) in enumerate(offs) if a >= resp_start and b > a]
        pos["resp_mean"] = r
        pos["resp_k8"] = r[:8]
    meta = {
        "trigger_aligned": bool(pos["trig"]),
        "trigger_method": method if pos["trig"] else "none",
        "n_tokens": len(ids),
        "n_prompt_tokens": len(pos["mean"]),
        "inst_last_token_id": ids[pos["inst_last"][0]] if pos["inst_last"] else None,
    }
    if item.get("spans"):
        meta["slots_aligned"] = bool(pos["slot_forma"]) and bool(pos["slot_intencao"])
    return Encoded(ids, pos, meta)


# ---------------------------------------------------------------- extração

def pad_right(batch: list[Encoded], pad_id: int):
    """ids e máscara [B, T] com padding à direita: o token t de cada item fica na coluna t."""
    T = max(len(e.input_ids) for e in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), T), dtype=torch.long)
    for i, e in enumerate(batch):
        ids[i, :len(e.input_ids)] = torch.tensor(e.input_ids)
        att[i, :len(e.input_ids)] = 1
    return ids, att


@torch.no_grad()
def run_batch(model, layers_mod, layers: list[int], batch: list[Encoded], pad_id: int):
    """Forward de um lote com padding à direita. Devolve {L: tensor [B, T, d]} (na GPU, bf16).

    O índice de token t de um item é a coluna t (sem deslocamento). Com padding à direita, os
    tokens reais têm as mesmas posições absolutas de quando o item roda sozinho, e a atenção
    causal impede que vejam o padding."""
    ids, att = pad_right(batch, pad_id)
    cap: dict[int, torch.Tensor] = {}
    hooks = []
    for L in layers:
        def hook(_m, _i, out, L=L):
            cap[L] = (out[0] if isinstance(out, tuple) else out)
        hooks.append(layers_mod[L].register_forward_hook(hook))
    try:
        dev = next(model.parameters()).device
        # o forward só precisa ir até a última camada pedida, mas o HF não corta: roda inteiro.
        model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return cap


def group_batches(encs: list[Encoded], items: list[dict], batch_size: int) -> list[list[int]]:
    """Lotes de índices, ordenados por tamanho, sem separar os itens de um mesmo `group_id`.

    Um grupo maior que `batch_size` vira um lote sozinho."""
    groups: dict = {}
    for i, it in enumerate(items):
        groups.setdefault(it.get("group_id", f"__item{i}"), []).append(i)
    ordered = sorted(groups.values(), key=lambda g: (max(len(encs[i].input_ids) for i in g), g[0]))
    batches, cur = [], []
    for g in ordered:
        if cur and len(cur) + len(g) > batch_size:
            batches.append(cur)
            cur = []
        cur = cur + g
    if cur:
        batches.append(cur)
    return batches


def _sparse_row(r: torch.Tensor | None):
    """Latente denso (GPU) -> (índices, valores) em CPU; None continua None."""
    if r is None:
        return None
    nz = torch.nonzero(r).flatten()
    return nz.cpu().numpy().astype(np.int32), r[nz].float().cpu().numpy()


def pool(h: torch.Tensor, idx: list[int]) -> torch.Tensor:
    """Média dos tokens `idx` de h [T, d] (padding à direita: índice do item = coluna)."""
    if not idx:
        return torch.full((h.shape[-1],), float("nan"), device=h.device)
    t = torch.tensor(idx, device=h.device)
    return h.index_select(0, t).float().mean(0)


def extract(model, tok, items: list[dict], layers: list[int] = LAYERS, batch_size: int = 16,
            saes: dict | None = None, positions: tuple[str, ...] | None = None,
            log_every: int = 20):
    """Extrai o cache denso (e latentes, se `saes`) de uma lista de itens.

    saes: {nome: {L: JumpReLUSAE}}; as variantes `_tokmean`/`_tokmax` são calculadas token a token.
    Devolve (dense {L: {pos: np.float32 [N, d]}}, latents {nome: {L: {pos: csr}}}, index [dict]).
    """
    import scipy.sparse as sp

    layers_mod, _ = get_layers(model)
    encs = [encode_item(tok, it) for it in items]
    has_resp = any("response" in it for it in items)
    if positions is None:
        has_spans = any(it.get("spans") for it in items)
        positions = BASE_POSITIONS + (SLOT_POSITIONS if has_spans else ("trig",)) \
            + (RESP_POSITIONS if has_resp else ())
    batches = group_batches(encs, items, batch_size)
    N = len(items)
    d = model.config.get_text_config().hidden_size
    dense = {L: {p: np.zeros((N, d), dtype=np.float32) for p in positions} for L in layers}
    lat_rows: dict = {}
    if saes:
        tok_pos = [p for p in positions if p in TOKEN_POOLED]
        lat_pos = list(positions) + [f"{p}_{a}" for p in tok_pos for a in ("tokmean", "tokmax")]
        lat_rows = {n: {L: {p: [None] * N for p in lat_pos} for L in s} for n, s in saes.items()}
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    t0 = time.time()
    done = 0
    for bi, idxs in enumerate(batches):
        batch = [encs[i] for i in idxs]
        cap = run_batch(model, layers_mod, layers, batch, pad_id)
        done += len(idxs)
        for L in layers:
            H = cap[L]
            for j, i in enumerate(idxs):
                e = encs[i]
                vecs = {p: pool(H[j], e.positions.get(p, [])) for p in positions}
                for p, v in vecs.items():
                    dense[L][p][i] = v.cpu().numpy()
                if not saes:
                    continue
                for name, per_layer in saes.items():
                    if L not in per_layer:
                        continue
                    sae = per_layer[L]
                    for p, v in vecs.items():
                        if torch.isnan(v).any():
                            lat_rows[name][L][p][i] = None
                        else:
                            lat_rows[name][L][p][i] = _sparse_row(sae.encode(v[None])[0])
                    for p in tok_pos:          # latentes token a token, depois média e máximo
                        tr = e.positions.get(p, [])
                        if tr:
                            f = sae.encode(H[j, tr].float())
                            lat_rows[name][L][f"{p}_tokmean"][i] = _sparse_row(f.mean(0))
                            lat_rows[name][L][f"{p}_tokmax"][i] = _sparse_row(f.amax(0))
                        else:
                            lat_rows[name][L][f"{p}_tokmean"][i] = None
                            lat_rows[name][L][f"{p}_tokmax"][i] = None
        if log_every and bi % log_every == 0:
            print(f"  lote {bi} ({done}/{N}) T={max(len(b.input_ids) for b in batch)} "
                  f"{time.time() - t0:.0f}s", flush=True)

    # float32, não float16: o residual do Gemma 3 passa de 65.504 (máx. do fp16) a partir de ~L17
    latents = {}
    for name, per_layer in lat_rows.items():
        latents[name] = {}
        for L, per_pos in per_layer.items():
            latents[name][L] = {}
            for p, rows in per_pos.items():
                d_sae = saes[name][L].d_sae
                data, indices, indptr = [], [], [0]
                for r in rows:
                    if r is not None:
                        indices.append(r[0])
                        data.append(r[1])
                        indptr.append(indptr[-1] + len(r[0]))
                    else:
                        indptr.append(indptr[-1])
                latents[name][L][p] = sp.csr_matrix(
                    (np.concatenate(data) if data else np.zeros(0, np.float32),
                     np.concatenate(indices) if indices else np.zeros(0, np.int64),
                     np.array(indptr)), shape=(len(rows), d_sae), dtype=np.float32)
    index = []
    for it, e in zip(items, encs):
        row = {k: it.get(k) for k in ("item_id", "group_id", "item", "label", "has_trigger",
                                      "split", "domain", "subdomain", "trigger", "trigger_type",
                                      "q_border_method", "template_id") if k in it}
        row.update(e.meta)
        index.append(row)
    return dense, latents, index


# ---------------------------------------------------------------- gravação

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_cache(root: Path, conjunto: str, dense, latents, index) -> list[Path]:
    import scipy.sparse as sp

    written = []
    for L, dp in dense.items():
        d = root / "resid_post" / f"L{L:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for p, a in dp.items():
            f = d / f"{conjunto}__{p}.npy"
            np.save(f, a)
            written.append(f)
    for name, per_layer in latents.items():
        for L, per_pos in per_layer.items():
            d = root / "resid_post" / f"L{L:02d}"
            d.mkdir(parents=True, exist_ok=True)
            for p, m in per_pos.items():
                f = d / f"{conjunto}__{p}__{name}.npz"
                sp.save_npz(f, m)
                written.append(f)
    (root / "index").mkdir(parents=True, exist_ok=True)
    f = root / "index" / f"{conjunto}.jsonl"
    with open(f, "w", encoding="utf-8") as fh:
        for r in index:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    written.append(f)
    return written


def update_manifest(root: Path, conjunto: str, written: list[Path], args: dict) -> Path:
    import transformers

    mf = root / "MANIFEST.json"
    man = json.loads(mf.read_text()) if mf.exists() else {"model_id": MODEL_ID, "sets": {}}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         cwd=Path(__file__).parent).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True,
                                             cwd=Path(__file__).parent).strip())
    except Exception:
        commit, dirty = None, None
    man["sets"][conjunto] = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": commit, "dirty": dirty, "args": args,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "files": {str(p.relative_to(root)): _sha256(p) for p in written},
    }
    mf.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    return mf


# ---------------------------------------------------------------- CLI

def load_set(name: str, limit: int | None = None) -> list[dict]:
    from featgrpo.interp import data

    if name == "pairs":
        items = data.prompt_items(data.load_pairs())
    elif name == "final_resp":
        items = []
        for s in data.SPLITS:
            for r in data.response_items(data.load_final(s)):
                items.append({**r, "text": r["prompt"]})
    elif name == "cells2x2":
        items = cells_items(data.load_cells_2x2())
    elif name == "controls":
        items = control_items(data._read_jsonl(data.PHASE1 / "controls_trig_only.jsonl"))
    else:
        p = data.PHASE1 / f"{name}.jsonl"
        items = data._read_jsonl(p)
    return items[:limit] if limit else items


def cells_items(groups: list[dict]) -> list[dict]:
    """Grupos do cells_2x2.jsonl -> um item por célula (`item` = nome da célula)."""
    out = []
    for g in groups:
        for cell in ("harm_trig", "harm_notrig", "ben_trig", "ben_notrig"):
            c = g["cells"][cell]
            # frente A: slots = {form, intent}; render_cells / docs: spans = {forma, intencao}
            sl = c.get("spans") or c.get("slots") or {}
            spans = {"forma": sl.get("forma", sl.get("form")),
                     "intencao": sl.get("intencao", sl.get("intent"))}
            out.append({
                "item_id": f"{g['group_id']}:{cell}", "group_id": g["group_id"], "item": cell,
                "label": int(cell.startswith("harm")), "has_trigger": int(cell.endswith("_trig")),
                "text": c["text"], "spans": spans, "trigger": g.get("trigger"),
                "trigger_type": g.get("trigger_type"), "split": g["split"], "domain": g["domain"],
                "subdomain": g.get("subdomain"),
                "q_border_method": (g.get("orig") or {}).get("q_border_method"),
            })
    return out


def control_items(rows: list[dict]) -> list[dict]:
    """controls_trig_only.jsonl -> itens `trig_only` / `neutral_only`; `trig` = a palavra da lacuna.

    Se a linha trouxer `spans_trig`/`spans_neutral` (de render_cells), usa-os; senão, busca a palavra."""
    out = []
    for k, r in enumerate(rows):
        gid = f"ctrl{k:05d}"
        for item, text, word, sk in (("trig_only", r["text_trig"], r["trigger"], "spans_trig"),
                                     ("neutral_only", r["text_neutral"], r["neutral_term"],
                                      "spans_neutral")):
            it = {"item_id": f"{gid}:{item}", "group_id": gid, "item": item,
                  "has_trigger": int(item == "trig_only"), "text": text, "trigger": word,
                  "template_id": r.get("template_id"), "split": r.get("split"),
                  "domain": r.get("domain")}
            if r.get(sk):
                it["spans"] = {"forma": r[sk]["forma"], "intencao": r[sk].get("intencao", r[sk]["forma"])}
            out.append(it)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="pairs | final_resp | <arquivo em datasets/phase1>")
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    ap.add_argument("--widths", nargs="*", default=["16k"], help="larguras do Gemma Scope 2")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=str(ACTS))
    args = ap.parse_args()
    set_seeds(0)

    from featgrpo.interp.common import load_model, load_sae

    items = load_set(args.set, args.limit)
    print(f"{args.set}: {len(items)} itens")
    model, tok = load_model()
    saes = {w: {L: load_sae(L, w) for L in args.layers} for w in args.widths}
    dense, latents, index = extract(model, tok, items, args.layers, args.batch_size, saes)
    root = Path(args.out_root)
    written = save_cache(root, args.set, dense, latents, index)
    mf = update_manifest(root, args.set, written, {
        **vars(args), "sae_ids": {w: {L: f"{SAE_REPO}/resid_post/{sae_id(L, w)}"
                                      for L in args.layers} for w in saes}})
    if any("slots_aligned" in r for r in index):
        na = sum(not r.get("slots_aligned", True) for r in index)
        msg = f"lacunas sem alinhamento: {na}"
    else:
        na = sum(not r["trigger_aligned"] for r in index)
        msg = f"sem alinhamento de gatilho: {na}"
    if any("has_trigger" in r for r in index):   # 2×2 e controles: mesmo token no inst_last?
        by_g: dict = {}
        for r in index:
            by_g.setdefault(r["group_id"], set()).add(r.get("inst_last_token_id"))
        diff = sum(len(v) > 1 for v in by_g.values())
        msg += f"; grupos com inst_last diferente entre células: {diff}/{len(by_g)}"
    print(f"ok: {len(written)} arquivos em {root}; manifest {mf}; {msg}")


if __name__ == "__main__":
    main()
