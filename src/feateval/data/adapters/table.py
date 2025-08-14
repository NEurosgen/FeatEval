import pandas as pd
import numpy as np
from typing import Tuple, List

def xy_from_table(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[feature_cols].copy()
    y = df[target_col].to_numpy()
    return X, y
