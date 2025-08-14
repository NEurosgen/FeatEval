# src/feateval/selectors/search_utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
from feateval.evaluate.set_level import eval_feature_set
from feateval.metrics.stats import bootstrap_diff

def delta_metric_ci(df: pd.DataFrame, target: str, cols_old: list[str], cols_new: list[str],
                    cv, estimator_ctor, metric_name: str = "pr_auc", n_boot: int = 2000, seed: int = 42):
    m_old = eval_feature_set(df, target, cols_old, cv, estimator_ctor)
    m_new = eval_feature_set(df, target, cols_new, cv, estimator_ctor)
    diff, lo, hi = bootstrap_diff(np.array(m_new), np.array(m_old), n_boot=n_boot, seed=seed)
    return diff, (lo, hi), m_old, m_new
