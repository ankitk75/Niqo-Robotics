import numpy as np

def max_pool_naive(img: np.ndarray, k: int) -> np.ndarray:
    m = img.shape[0]
    out = np.empty((m - k + 1, m - k + 1), dtype=img.dtype)
    for i in range(m - k + 1):
        for j in range(m - k + 1):
            out[i, j] = img[i:i + k, j:j + k].max()
    return out
