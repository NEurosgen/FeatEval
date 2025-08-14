# src/feateval/selectors/search.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
from .search_utils import delta_metric_ci

def sffs_beam_search(df: pd.DataFrame, target: str,
                     start_cols: List[str], pool: List[str],
                     cv, estimator_ctor,
                     beam: int = 4, patience: int = 3,
                     top_T: int = 32, n_boot: int = 1000, seed: int = 42) -> Tuple[List[str], list[dict]]:

    rng = np.random.default_rng(seed)
    Beam = [list(start_cols)]
    best_cols = list(start_cols)
    best_delta = 0.0
    no_imp = 0
    trace: list[dict] = []

    pool = [c for c in pool if c not in start_cols]

    while True:
        trial_feats = pool[:top_T] if len(pool) > top_T else pool
        candidates = []

        for S in Beam:
            for f in trial_feats:
                if f in S:
                    continue
                S_new = S + [f]
                delta, (lo, hi), _, _ = delta_metric_ci(df, target, S, S_new, cv, estimator_ctor,
                                                        n_boot=n_boot, seed=int(rng.integers(0, 10**9)))
                candidates.append((delta, lo, hi, S_new, f))

        ok = [c for c in candidates if c[1] > 0]  

        for d, lo, hi, S_new, f in sorted(candidates, key=lambda t: t[0], reverse=True)[:max(beam, 5)]:
            trace.append({"add": f, "delta": d, "lo": lo, "hi": hi, "k": len(S_new)})

        if not ok:
            no_imp += 1
        else:
            no_imp = 0
            best_cand = max(ok, key=lambda t: t[0])  
            if best_cand[0] > best_delta:
                best_delta = best_cand[0]
                best_cols = list(best_cand[3])


        Beam = [c[3] for c in sorted(candidates, key=lambda t: t[0], reverse=True)[:beam]]

        if no_imp >= patience or len(Beam) == 0:
            break

    return best_cols, trace
