import numpy as np

def model_delta(w_new, w_old):
    if w_old is None:
        return float("inf")
    num = sum(np.linalg.norm(a - b) for a, b in zip(w_new, w_old))
    den = sum(np.linalg.norm(b) for b in w_old) + 1e-8
    return num / den

