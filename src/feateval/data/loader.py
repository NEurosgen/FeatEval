from .schema import FeatureSpec
from .registry_init import reg 
from .validate import validate_shapes

def load_group(spec: FeatureSpec, name: str):
    X, y = reg.get(name)           
    validate_shapes(X, y)          
    return X, y

def load_union(spec: FeatureSpec, names: list[str]):

    pairs = [reg.get(n) for n in names]
    ys = [p[1] for p in pairs]
    import numpy as np
    Xs = [p[0] for p in pairs]
    X = np.concatenate(Xs, axis=1) if isinstance(Xs[0], np.ndarray) else ...
    return X, ys[0]
