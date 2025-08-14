import numpy as np
import pandas as pd
from src.feateval.data.splits import CVConfig, CVEngine
from src.feateval.evaluate.set_level import eval_feature_set
from src.feateval.models.sk_linear import SKLogReg

def test_eval_feature_set_runs():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a1": rng.normal(size=120),
        "a2": rng.normal(size=120),
        "b1": rng.normal(size=120),
        "label": rng.integers(0,2,size=120),
    })
    cv = CVEngine(CVConfig(cv_type="stratified", n_splits=5, shuffle=True, seed=7))
    sA = eval_feature_set(df, "label", ["a1","a2"], cv, SKLogReg)
    sB = eval_feature_set(df, "label", ["b1"], cv, SKLogReg)
    assert len(sA) == 5 and len(sB) == 5
