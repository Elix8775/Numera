# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

def dot(double[:, :] A, double[:, :] B):
    cdef int n = A.shape[0]
    cdef int m = B.shape[1]
    cdef int k = A.shape[1]
    cdef int i, j, l

    if k != B.shape[0]:
        raise ValueError(f"Dimensions incompatibles : ({n}x{k}) @ ({B.shape[0]}x{m})")

    cdef np.ndarray[double, ndim=2] C = np.zeros((n, m), dtype=np.float64)
    cdef double[:, :] Cv = C

    for i in range(n):
        for j in range(m):
            for l in range(k):
                Cv[i, j] += A[i, l] * B[l, j]
    return C

def determinant(double[:, :] A):
    cdef int n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("La matrice doit être carrée")
    if n == 1:
        return A[0, 0]
    if n == 2:
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

    # Pour n > 2 : décomposition LU via NumPy (délégué)
    return np.linalg.det(np.asarray(A))

def inverse(double[:, :] A):
    cdef int n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("La matrice doit être carrée")
    cdef double d = determinant(A)
    if d == 0:
        raise ValueError("Matrice singulière, non inversible")
    return np.linalg.inv(np.asarray(A))

def eigen(double[:, :] A):
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matrice doit être carrée")
    eigenvals, eigenvecs = np.linalg.eig(np.asarray(A))
    return eigenvals, eigenvecs