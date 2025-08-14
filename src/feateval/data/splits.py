
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
class CVEngine:
    def __init__(self, cv_type: str, n_splits: int = 5, n_repeats: int = 1, seed: int = 42, groups=None):
        self.cv_type = cv_type
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.seed = seed
        self.groups = groups
    def split(self, X, y):
        cv = None
        if self.cv_type == 'stratifiedkfold':
            cv = StratifiedKFold( n_splits=self.n_splits,
            shuffle=False,
            random_state=self.cv.seed,
            )
       
        return  list(cv.split(X, y))
    
    def save(self, path: str):
        with open(path,'wb') as f:
            pickle.dump(self.split,f)
    
    @classmethod
    def load( path: str):
        with open(path, 'rb') as f:
            obj = pickle.load(f) 
