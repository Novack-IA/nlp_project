# Pedidos para a frente B

Formato: `- [ ] AAAA-MM-DD · quem pede · o que precisa · para quê`

- [ ] 2026-11-20 · Davi · Confirmar (ou corrigir) a convenção que `src/featgrpo/interp/cache_io.py` assume: latentes em `resid_post/L{L:02d}/{conjunto}__{posicao}__{sae}.npz`, salvos com `scipy.sparse.save_npz` (CSR), com `sae` = `16k`, `262k` ou `proprio` · para `classifiers.py` ler os latentes sem adaptação. Se for outro formato, basta mudar `latents_path`/`load_latents`.
- [ ] 2026-11-20 · Davi · No `index/cells2x2.jsonl`, `item` = nome da célula (`harm_trig`, …), mais `n_tokens` de cada célula · os probes e classificadores separam as células por `item`, e o recorte C4c usa `n_tokens`. Para os originais do 2×2, se entrarem no mesmo conjunto, use `item` = `q_harm`/`q_border`.

