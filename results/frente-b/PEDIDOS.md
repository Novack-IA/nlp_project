# Pedidos para a frente B

Formato: `- [ ] AAAA-MM-DD · quem pede · o que precisa · para quê`

- [ ] 2026-11-20 · Davi · Confirmar o **nome** dos arquivos de latentes que `src/featgrpo/interp/cache_io.py` assume: `resid_post/L{L:02d}/{conjunto}__{posicao}__{sae}.npz`, com `sae` = `16k`, `262k` ou `proprio` · para `classifiers.py` achar os arquivos. **O formato interno já não importa:** o leitor aceita tanto `scipy.sparse.save_npz` quanto o CSR "à mão" do B2 (`indices`, `values`, `indptr`), desde que o CSR à mão traga `shape` ou `d_sae`.
- [ ] 2026-11-20 · Davi · As posições das lacunas no `cells_2x2.jsonl` real (código da frente A) ficam em `cells[c].slots.form` e `cells[c].slots.intent`, e não em `cells[c].spans.forma`/`intencao`, como está escrito em `caio-frente-b.md` B2 · use os nomes do arquivo real no extrator; o validador da frente D aceita os dois.
- [ ] 2026-11-20 · Davi · No `index/cells2x2.jsonl`, `item` = nome da célula (`harm_trig`, …), mais `n_tokens` de cada célula · os probes e classificadores separam as células por `item`, e o recorte C4c usa `n_tokens`. Para os originais do 2×2, se entrarem no mesmo conjunto, use `item` = `q_harm`/`q_border`.

