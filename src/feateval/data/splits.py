import numpy as np
import pickle
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Literal
from sklearn.model_selection import (
    StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit
)

CVType = Literal["stratified", "kfold", "group", "time"]

@dataclass
class CVConfig:
    cv_type: CVType = "stratified"
    n_splits: int = 5
    n_repeats: int = 1
    shuffle: bool = True
    seed: int = 42

class CVEngine:
    def __init__(self, config: CVConfig):
        self.cfg = config
        self._splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

    def _base_cv(self, y=None, groups=None):
        c = self.cfg
        if c.cv_type == "stratified":
            return StratifiedKFold(
                n_splits=c.n_splits,
                shuffle=c.shuffle,
                random_state=c.seed if c.shuffle else None,
            )
        elif c.cv_type == "kfold":
            return KFold(
                n_splits=c.n_splits,
                shuffle=c.shuffle,
                random_state=c.seed if c.shuffle else None,
            )
        elif c.cv_type == "group":
            return GroupKFold(n_splits=c.n_splits)
        elif c.cv_type == "time":
            return TimeSeriesSplit(n_splits=c.n_splits)
        else:
            raise ValueError(f"Unknown cv_type: {c.cv_type}")

    def split(
        self,
        X,
        y,
        groups: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self._splits is not None:
            return self._splits

        c = self.cfg
        splits: List[Tuple[np.ndarray, np.ndarray]] = []

        
        for r in range(c.n_repeats):
            seed_r = c.seed + r
            if c.cv_type in ("stratified", "kfold"):
                cv = type(self._base_cv(y))(
                    n_splits=c.n_splits,
                    shuffle=c.shuffle,
                    random_state=seed_r if c.shuffle else None
                )
            else:
                cv = self._base_cv(y, groups)

            if c.cv_type == "group":
                if groups is None:
                    raise ValueError("groups must be provided for group CV")
                iterator = cv.split(X, y, groups=groups)
            elif c.cv_type == "stratified":
                iterator = cv.split(X, y)
            elif c.cv_type == "kfold":
                iterator = cv.split(X)
            elif c.cv_type == "time":
                iterator = cv.split(X)
            else:
                raise AssertionError

            for tr, va in iterator:
                splits.append((np.asarray(tr), np.asarray(va)))

        self._splits = splits
        return splits

    def save(self, path: str) -> None:
        payload = {
            "config": self.cfg,
            "splits": self._splits
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "CVEngine":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        eng = cls(payload["config"])
        eng._splits = payload["splits"]
        return eng
