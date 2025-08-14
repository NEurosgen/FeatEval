def validate_shapes(X, y):

    n = X.shape[0] if hasattr(X, "shape") else len(X)
    assert len(y) == n, f"len(y)={len(y)} != n_samples={n}"
    import numpy as np
    assert not np.isnan(y).any(), "y contains NaN"
