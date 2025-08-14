import numpy as np, pandas as pd
from src.feateval.data.splits import CVConfig, CVEngine
from src.feateval.selectors.filters import rank_by_mi
from src.feateval.selectors.search import sffs_beam_search
from src.feateval.models.sk_linear import SKLogReg

def test_sffs_beam_smoke():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "a1": rng.normal(size=n),
        "a2": rng.normal(size=n),
        "b1": rng.normal(size=n),
        "b2": rng.normal(size=n),
        "label": rng.integers(0,2,size=n),
    })
    cv = CVEngine(CVConfig(cv_type="stratified", n_splits=4, shuffle=True, seed=1))
    A = ["a1","a2"]; B = ["b1","b2"]
    pool = sorted(set(A+B))
    ranked = rank_by_mi(df[pool], df["label"].to_numpy(), pool)
    S0 = A
    S, trace = sffs_beam_search(df, "label", S0, [c for c in ranked if c not in S0], cv, SKLogReg,
                                beam=3, patience=2, top_T=3, n_boot=100)  # маленький n_boot = быстрее
    assert isinstance(S, list) and len(S) >= len(S0)
