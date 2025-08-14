
from __future__ import annotations
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from .search import sffs_beam_search

def stability_selection(df: pd.DataFrame, target: str,
                        start_cols: List[str], pool: List[str],
                        cv_factory, estimator_ctor,
                        repeats: int = 20, threshold: float = 0.7,
                        **search_kwargs) -> Tuple[List[str], Dict[str, float]]:

    counts = Counter()
    for r in range(repeats):
        cv = cv_factory(r)  
        S, _ = sffs_beam_search(df, target, start_cols, pool, cv, estimator_ctor, **search_kwargs)
        counts.update(S)
    freq = {f: counts[f] / repeats for f in counts}
    S_star = [f for f, p in freq.items() if p >= threshold]
    return S_star, freq
