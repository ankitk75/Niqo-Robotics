import sys
import time
import numpy as np
from max_pool_naive import max_pool_naive
from max_pool_fast import max_pool_fast

try:
    M = int(input("Enter image size (m): "))
    K = int(input("Enter window size (k): "))
except ValueError:
    print("Numbers only, please.")
    sys.exit(1)

rng = np.random.default_rng(0)
image = rng.integers(0, 1000, size=(M, M))

t0 = time.perf_counter()
out_naive = max_pool_naive(image, K)
naive_time = time.perf_counter() - t0
print(f"naïve  : {naive_time:.4f} s")

t0 = time.perf_counter()
out_fast = max_pool_fast(image, K)
fast_time = time.perf_counter() - t0
speedup = float("inf") if fast_time == 0 else naive_time / fast_time
print(f"fast   : {fast_time:.4f} s (×{speedup:.1f} faster)")

print("outputs match ✔" if np.array_equal(out_naive, out_fast) else "outputs differ ✘")
