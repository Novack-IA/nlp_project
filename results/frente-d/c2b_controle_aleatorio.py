"""Controle do C2(b): queda no XSTest de features aleatórias (mesmo tamanho) e do denso, mesma régua do TF-IDF."""
import json, sys
from pathlib import Path
import numpy as np
from featgrpo.interp import cache_io, classifiers as C
from featgrpo.interp.probes import failed_groups, subset_mask

root = Path("artifacts/acts/gemma-3-4b-it"); L, pos = 29, "post_last"
out = {}
for sae in sys.argv[1:]:
    rows = cache_io.load_index(root, "cells2x2")
    keep = np.array([r["cell"] is not None for r in rows]) & subset_mask(rows, "principal", failed_groups())
    rows = [r for r, k in zip(rows, keep) if k]
    Zs = cache_io.load_latents(root, "cells2x2", L, pos, sae)[keep]
    s_int = C.load_list(Path(f"artifacts/_downloads/listas_c/S_intent_L{L}_{pos}_{sae}.json"))
    tr = np.array([r["split"] == "train" for r in rows])
    freq = np.asarray((Zs[tr] > 0).mean(0)).ravel()
    pool = np.flatnonzero(freq >= C.MIN_FREQ)
    texts, _ = C.load_cells_texts()
    Zx, xs_rows, xs_texts = C.load_xstest(root, L, pos, sae)
    texts_of = lambda r: texts[(r["group_id"], r["cell"])]
    idx = lambda f: np.flatnonzero([f(r) for r in rows])
    def drop(cols, Ztr=None, Zxx=None):
        Z = Zs[:, cols].toarray().astype(np.float32) if Ztr is None else Ztr
        X = Zx[:, cols] if Zxx is None else Zxx
        f = C._fold(Z, texts_of, rows, idx(lambda r: r["split"] == "train"), idx(lambda r: r["split"] == "test"),
                    None, out_Z=X, out_rows=xs_rows, out_texts=xs_texts)
        return C.relative_drop_folds([f], 200)
    res = {"S_intent": drop(np.array(s_int))}
    rng = np.random.default_rng(0); draws = []
    in_pool = np.setdiff1d(pool, s_int)
    for i in range(20):
        d = drop(rng.choice(in_pool, size=len(s_int), replace=False))
        draws.append({k: d.get(k) for k in ("queda_features", "razao", "atende")})
    res["aleatorio_20"] = {"queda_features": [x["queda_features"] for x in draws], "razao": [x["razao"] for x in draws],
                           "n_atende": sum(bool(x["atende"]) for x in draws)}
    out[sae] = res
    print(sae, json.dumps({"S_intent": {k: res["S_intent"].get(k) for k in ("queda_features", "queda_tfidf", "razao", "ic95")},
                           "aleatorio": {"queda_mediana": float(np.median(res["aleatorio_20"]["queda_features"])),
                                         "queda_min_max": [min(res["aleatorio_20"]["queda_features"]), max(res["aleatorio_20"]["queda_features"])],
                                         "razao_mediana": float(np.median([r for r in res["aleatorio_20"]["razao"] if r is not None])),
                                         "n_atende": res["aleatorio_20"]["n_atende"]}}), flush=True)
# denso
Xd = cache_io.load_dense(root, "cells2x2", L, pos)[keep]
xr = cache_io.load_index(root, "xstest")
Xxd = cache_io.load_dense(root, "xstest", L, pos)
ids = [r["group_id"] for r in xs_rows]; pos_of = {r["group_id"]: i for i, r in enumerate(xr)}
Xxd = Xxd[[pos_of[g] for g in ids]]
dd = drop(None, Ztr=Xd, Zxx=Xxd)
out["denso"] = dd
print("denso", json.dumps({k: dd.get(k) for k in ("queda_features", "queda_tfidf", "razao", "ic95")}))
Path("results/frente-d/c2b_controle_aleatorio.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
