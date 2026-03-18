import numpy as np

def destroy(n):
    a = np.zeros((n, n), dtype=complex)
    for k in range(1, n):
        a[k-1, k] = np.sqrt(k)
    return a