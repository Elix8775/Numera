# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow

def mean(double[:] data):
    cdef int n = data.shape[0]
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += data[i]
    return total / n

def median(double[:] data):
    cdef np.ndarray sorted_data = np.sort(np.asarray(data))
    cdef int n = sorted_data.shape[0]
    if n % 2 == 1:
        return sorted_data[n // 2]
    return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0

def mode(double[:] data):
    values, counts = np.unique(np.asarray(data), return_counts=True)
    return values[np.argmax(counts)]

def variance(double[:] data, int ddof=0):
    cdef int n = data.shape[0]
    cdef double m = mean(data)
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += (data[i] - m) ** 2
    return total / (n - ddof)

def std(double[:] data, int ddof=0):
    return sqrt(variance(data, ddof))

def data_range(double[:] data):
    return np.max(np.asarray(data)) - np.min(np.asarray(data))

def iqr(double[:] data):
    cdef np.ndarray arr = np.sort(np.asarray(data))
    return np.percentile(arr, 75) - np.percentile(arr, 25)

def skewness(double[:] data):
    cdef int n = data.shape[0]
    cdef double m = mean(data)
    cdef double s = std(data)
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += pow((data[i] - m) / s, 3)
    return total / n

def kurtosis(double[:] data):
    cdef int n = data.shape[0]
    cdef double m = mean(data)
    cdef double s = std(data)
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += pow((data[i] - m) / s, 4)
    return (total / n) - 3.0   # kurtosis excédentaire (= 0 pour loi normale)

def percentile(double[:] data, double p):
    if not (0 <= p <= 100):
        raise ValueError("p doit être entre 0 et 100")
    return np.percentile(np.asarray(data), p)

def covariance(double[:] x, double[:] y):
    cdef int n = x.shape[0]
    cdef double mx = mean(x)
    cdef double my = mean(y)
    cdef double total = 0.0
    cdef int i
    for i in range(n):
        total += (x[i] - mx) * (y[i] - my)
    return total / n

def correlation(double[:] x, double[:] y):
    return covariance(x, y) / (std(x) * std(y))