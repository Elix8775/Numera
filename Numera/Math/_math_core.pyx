# Numera/Math/_math_core.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport log, log2, log10, exp, pow, sqrt, fabs
from libc.stdint cimport int64_t

# ── Logarithmes ────────────────────────────────────────────────
def ln(double x):
    if x <= 0: raise ValueError("ln défini uniquement pour x > 0")
    return log(x)

def log_b(double x, double base=10.0):
    if x <= 0: raise ValueError("log défini uniquement pour x > 0")
    if base <= 0 or base == 1: raise ValueError("base invalide")
    return log(x) / log(base)

def log2_f(double x):
    if x <= 0: raise ValueError("log2 défini uniquement pour x > 0")
    return log2(x)

def log10_f(double x):
    if x <= 0: raise ValueError("log10 défini uniquement pour x > 0")
    return log10(x)

def log_array(double[:] arr):
    cdef int n = arr.shape[0]
    cdef np.ndarray[double] out = np.empty(n, dtype=np.float64)
    cdef int i
    for i in range(n):
        if arr[i] <= 0: raise ValueError(f"Valeur non positive à l'indice {i}")
        out[i] = log(arr[i])
    return out

# ── Exponentielles & puissances ────────────────────────────────
def exp_f(double x):
    return exp(x)

def exp_array(double[:] arr):
    cdef int n = arr.shape[0]
    cdef np.ndarray[double] out = np.empty(n, dtype=np.float64)
    cdef int i
    for i in range(n):
        out[i] = exp(arr[i])
    return out

def pow_f(double x, double p):
    return pow(x, p)

def sqrt_f(double x):
    if x < 0: raise ValueError("sqrt défini uniquement pour x >= 0")
    return sqrt(x)

def sqrt_array(double[:] arr):
    cdef int n = arr.shape[0]
    cdef np.ndarray[double] out = np.empty(n, dtype=np.float64)
    cdef int i
    for i in range(n):
        if arr[i] < 0: raise ValueError(f"Valeur négative à l'indice {i}")
        out[i] = sqrt(arr[i])
    return out

# ── Sommes & produits ──────────────────────────────────────────
def sum_f(double[:] arr):
    cdef int n = arr.shape[0]
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += arr[i]
    return total

def product(double[:] arr):
    cdef int n = arr.shape[0]
    cdef double total = 1.0
    cdef int i
    for i in range(n):
        total *= arr[i]
    return total

def cumsum(double[:] arr):
    cdef int n = arr.shape[0]
    cdef np.ndarray[double] out = np.empty(n, dtype=np.float64)
    cdef double acc = 0.0
    cdef int i
    for i in range(n):
        acc += arr[i]
        out[i] = acc
    return out

def cumprod(double[:] arr):
    cdef int n = arr.shape[0]
    cdef np.ndarray[double] out = np.empty(n, dtype=np.float64)
    cdef double acc = 1.0
    cdef int i
    for i in range(n):
        acc *= arr[i]
        out[i] = acc
    return out

def diff(double[:] arr):
    """Différences successives : [a1-a0, a2-a1, ...]"""
    cdef int n = arr.shape[0]
    cdef np.ndarray[double] out = np.empty(n - 1, dtype=np.float64)
    cdef int i
    for i in range(n - 1):
        out[i] = arr[i + 1] - arr[i]
    return out

# ── Factorielle, combinaisons, permutations ────────────────────
def factorial(int64_t n):
    if n < 0: raise ValueError("Factorielle définie pour n >= 0")
    if n == 0 or n == 1: return 1
    cdef int64_t result = 1
    cdef int64_t i
    for i in range(2, n + 1):
        result *= i
    return result

def combination(int64_t n, int64_t k):
    """C(n, k) = n! / (k! * (n-k)!)"""
    if k < 0 or k > n: raise ValueError("Invalide : besoin de 0 <= k <= n")
    if k == 0 or k == n: return 1
    if k > n - k: k = n - k   # optimisation
    cdef int64_t result = 1
    cdef int64_t i
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

def permutation(int64_t n, int64_t k):
    """P(n, k) = n! / (n-k)!"""
    if k < 0 or k > n: raise ValueError("Invalide : besoin de 0 <= k <= n")
    cdef int64_t result = 1
    cdef int64_t i
    for i in range(n, n - k, -1):
        result *= i
    return result