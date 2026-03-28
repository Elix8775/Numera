import numpy as np
from . import _stats_core as _core

class Stats:
    def __init__(self, data):
        self._data = np.array(data, dtype=np.float64).flatten()
        if self._data.size == 0:
            raise ValueError("Les données ne peuvent pas être vides")

    def mean(self):
        """Moyenne arithmétique"""
        return _core.mean(self._data)

    def median(self):
        """Médiane"""
        return _core.median(self._data)

    def mode(self):
        """Valeur la plus fréquente"""
        return _core.mode(self._data)

    def variance(self, ddof=0):
        """Variance  (ddof=0 → population, ddof=1 → échantillon)"""
        return _core.variance(self._data, ddof)

    def std(self, ddof=0):
        """Écart-type"""
        return _core.std(self._data, ddof)

    def range(self):
        """Étendue (max - min)"""
        return _core.data_range(self._data)

    def iqr(self):
        """Intervalle interquartile (Q3 - Q1)"""
        return _core.iqr(self._data)

    def skewness(self):
        """Asymétrie de la distribution"""
        return _core.skewness(self._data)

    def kurtosis(self):
        """Aplatissement de la distribution"""
        return _core.kurtosis(self._data)

    def percentile(self, p):
        """Percentile p (entre 0 et 100)"""
        return _core.percentile(self._data, p)

    @staticmethod
    def covariance(x, y):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        if len(x) != len(y):
            raise ValueError("x et y doivent avoir la même taille")
        return _core.covariance(x, y)

    @staticmethod
    def correlation(x, y):
        """Coefficient de Pearson"""
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        return _core.correlation(x, y)

    def __repr__(self):
        return (f"Stats(n={len(self._data)}, "
                f"mean={self.mean():.4f}, "
                f"std={self.std():.4f})")