
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from typing import List

def rank_by_mi(X: pd.DataFrame, y: np.ndarray, candidates: List[str], n_neighbors: int = 3) -> List[str]:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas.DataFrame for rank_by_mi")
    num = X[candidates].select_dtypes("number")
    if num.shape[1] == 0:
        return []
    med = num.median()
    Xc = num.fillna(med).to_numpy()
    mi = mutual_info_classif(Xc, y, n_neighbors=n_neighbors, random_state=42, discrete_features=False)
    order = np.argsort(-mi)
    return [num.columns[i] for i in order]
