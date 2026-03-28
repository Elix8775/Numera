import numpy as np
from . import _matrix_core as _core

class Matrix:
    def __init__(self, data):
        self._data = np.array(data, dtype=np.float64)
        if self._data.ndim != 2:
            raise ValueError("Une matrice doit être 2D")

    @property
    def shape(self):
        return self._data.shape

    @property
    def T(self):
        return Matrix(self._data.T)

    def __repr__(self):
        return f"Matrix({self._data})"

    def __add__(self, other):
        return Matrix(self._data + other._data)

    def __sub__(self, other):
        return Matrix(self._data - other._data)

    def __mul__(self, other):
        return Matrix(self._data * other._data)

    def __matmul__(self, other):
        return Matrix(_core.dot(self._data, other._data))

    def det(self):
        return _core.determinant(self._data)

    def inv(self):
        return Matrix(_core.inverse(self._data))

    def eigenvalues(self):
        return _core.eigen(self._data)