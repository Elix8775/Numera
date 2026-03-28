# Numera/Math/functions.py
import numpy as np
from . import _math_core as _core

class Math:

    # ── Logarithmes ────────────────────────────────────────────
    @staticmethod
    def ln(x):
        """Logarithme naturel (base e)"""
        if hasattr(x, '__len__'):
            return _core.log_array(np.array(x, dtype=np.float64))
        return _core.ln(x)

    @staticmethod
    def log(x, base=10.0):
        """Logarithme en base quelconque (défaut base 10)"""
        return _core.log_b(x, base)

    @staticmethod
    def log2(x):
        return _core.log2_f(x)

    @staticmethod
    def log10(x):
        return _core.log10_f(x)

    # ── Exponentielles & puissances ────────────────────────────
    @staticmethod
    def exp(x):
        """e^x — accepte un scalaire ou une liste"""
        if hasattr(x, '__len__'):
            return _core.exp_array(np.array(x, dtype=np.float64))
        return _core.exp_f(x)

    @staticmethod
    def pow(x, p):
        return _core.pow_f(x, p)

    @staticmethod
    def sqrt(x):
        """Racine carrée — accepte un scalaire ou une liste"""
        if hasattr(x, '__len__'):
            return _core.sqrt_array(np.array(x, dtype=np.float64))
        return _core.sqrt_f(x)

    # ── Sommes & produits ──────────────────────────────────────
    @staticmethod
    def sum(arr):
        return _core.sum_f(np.array(arr, dtype=np.float64))

    @staticmethod
    def product(arr):
        return _core.product(np.array(arr, dtype=np.float64))

    @staticmethod
    def cumsum(arr):
        return _core.cumsum(np.array(arr, dtype=np.float64))

    @staticmethod
    def cumprod(arr):
        return _core.cumprod(np.array(arr, dtype=np.float64))

    @staticmethod
    def diff(arr):
        """Différences successives"""
        return _core.diff(np.array(arr, dtype=np.float64))

    # ── Combinatoire ───────────────────────────────────────────
    @staticmethod
    def factorial(n):
        """n! — ex: factorial(5) = 120"""
        return _core.factorial(n)

    @staticmethod
    def C(n, k):
        """Combinaisons C(n,k) — ex: C(5,2) = 10"""
        return _core.combination(n, k)

    @staticmethod
    def P(n, k):
        """Permutations P(n,k) — ex: P(5,2) = 20"""
        return _core.permutation(n, k)