import numpy as np, pandas as pd
from src.feateval.data.splits import CVConfig, CVEngine
from src.feateval.evaluate.compare_sets import compare_sets
from src.feateval.models.sk_linear import SKLogReg

def test_compare_sets_returns_diffs():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "a1": rng.normal(size=150),
        "a2": rng.normal(size=150),
        "b1": rng.normal(size=150),
        "label": rng.integers(0,2,size=150),
    })
    cv = CVEngine(CVConfig(cv_type="stratified", n_splits=5, shuffle=True, seed=42))
    out = compare_sets(df, "label", ["a1","a2"], ["b1"], cv, SKLogReg)
    assert "diff" in out and "A-B" in out["diff"]
    lo, hi = out["diff"]["A-B"][1], out["diff"]["A-B"][2]
    assert lo <= hi
