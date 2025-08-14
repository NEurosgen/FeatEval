import numpy as np
def bootstrap_diff(a, b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    samples = diff[idx].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)
