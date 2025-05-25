import numpy as np
from collections import deque

def _sliding_max_1d(arr: np.ndarray, k: int) -> np.ndarray:
    dq = deque()
    out = []
    for i, x in enumerate(arr):
        while dq and arr[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(arr[dq[0]])
    return np.array(out, dtype=arr.dtype)

def max_pool_fast(img: np.ndarray, k: int) -> np.ndarray:
    m = img.shape[0]
    inner = m - k + 1

    # horizontal pass
    row_max = np.empty((m, inner), dtype=img.dtype)
    for r in range(m):
        row_max[r] = _sliding_max_1d(img[r], k)

    # vertical pass
    out = np.empty((inner, inner), dtype=img.dtype)
    for c in range(inner):
        out[:, c] = _sliding_max_1d(row_max[:, c], k)
    return out
