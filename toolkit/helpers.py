import numpy as np

"""
Create the annihilation operator, in the number basis.

nlevels: number of levels
"""
def destroy(nlevels):
    a = np.zeros((nlevels, nlevels), dtype=complex)
    for k in range(1, nlevels):
        a[k-1, k] = np.sqrt(k)
    return a

# Pauli matrices and 2x2 identity
px = np.array([[0, 1], [1, 0]], dtype=complex)
py = np.array([[0, -1j], [1j, 0]], dtype=complex)
pz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)