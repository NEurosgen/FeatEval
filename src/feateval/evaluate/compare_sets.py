import numpy as np
from .set_level import eval_feature_set
from ..metrics.stats import bootstrap_diff

def compare_sets(df, target, A_cols, B_cols, cv, estimator_ctor):
    U_cols = sorted(set(A_cols) | set(B_cols))
    mA = eval_feature_set(df, target, A_cols, cv, estimator_ctor)
    mB = eval_feature_set(df, target, B_cols, cv, estimator_ctor)
    mU = eval_feature_set(df, target, U_cols, cv, estimator_ctor)
    d_AB = bootstrap_diff(np.array(mA), np.array(mB))
    d_UA = bootstrap_diff(np.array(mU), np.array(mA))
    d_UB = bootstrap_diff(np.array(mU), np.array(mB))
    return {"A": mA, "B": mB, "U": mU, "diff": {"A-B": d_AB, "U-A": d_UA, "U-B": d_UB}}
