# Numera/Calculus/_calculus_core.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt

# ── Dérivée (différences finies centrées) ─────────────────────
def derivative(f, double x, double h=1e-5):
    return (f(x + h) - f(x - h)) / (2.0 * h)

# ── Dérivée d'ordre n ─────────────────────────────────────────
def derivative_n(f, double x, int n=2, double h=1e-5):
    if n == 1:
        return derivative(f, x, h)
    # Formule récursive
    return (derivative_n(f, x + h, n - 1, h) - derivative_n(f, x - h, n - 1, h)) / (2.0 * h)

# ── Gradient ──────────────────────────────────────────────────
def gradient(f, double[:] point, double h=1e-5):
    cdef int n = point.shape[0]
    cdef np.ndarray[double] grad = np.zeros(n, dtype=np.float64)
    cdef np.ndarray[double] p_plus, p_minus
    cdef int i

    for i in range(n):
        p_plus  = np.array(point)
        p_minus = np.array(point)
        p_plus[i]  += h
        p_minus[i] -= h
        grad[i] = (f(*p_plus) - f(*p_minus)) / (2.0 * h)
    return grad

# ── Intégrale — Trapèzes ──────────────────────────────────────
def integrate_trapeze(f, double a, double b, int n=1000):
    cdef double h = (b - a) / n
    cdef double total = 0.5 * (f(a) + f(b))
    cdef int i
    cdef double x

    for i in range(1, n):
        x = a + i * h
        total += f(x)
    return total * h

# ── Intégrale — Simpson ───────────────────────────────────────
def integrate_simpson(f, double a, double b, int n=1000):
    cdef double h = (b - a) / n
    cdef double total = f(a) + f(b)
    cdef int i
    cdef double x

    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            total += 2.0 * f(x)
        else:
            total += 4.0 * f(x)
    return total * h / 3.0

# ── Intégrale — Romberg ───────────────────────────────────────
def integrate_romberg(f, double a, double b, double tol=1e-8):
    cdef int max_steps = 20
    cdef list R = []
    cdef double h, val
    cdef int i, j, k, n, last

    R.append([0.5 * (b - a) * (f(a) + f(b))])

    for i in range(1, max_steps):
        n = <int>(2 ** i)          # ← cast explicite en int
        h = (b - a) / n
        val = 0.0
        for k in range(1, n, 2):
            val += f(a + k * h)
        R.append([0.5 * R[i-1][0] + h * val])

        for j in range(1, i + 1):
            factor = 4.0 ** j
            R[i].append((factor * R[i][j-1] - R[i-1][j-1]) / (factor - 1.0))

        last = len(R[i]) - 1       # ← remplace l'indice négatif -1
        prev = len(R[i-1]) - 1
        if fabs(R[i][last] - R[i-1][prev]) < tol:
            return R[i][last]

    last = len(R[-1]) - 1          # ← idem ici
    return R[<int>(len(R)-1)][last]

# ── ODE — Méthode d'Euler ─────────────────────────────────────
def euler(f, double y0, double t0, double tf, double dt=0.01):
    cdef int n = int((tf - t0) / dt) + 1
    cdef np.ndarray[double] t = np.linspace(t0, tf, n)
    cdef np.ndarray[double] y = np.zeros(n, dtype=np.float64)
    cdef int i

    y[0] = y0
    for i in range(1, n):
        y[i] = y[i-1] + dt * f(t[i-1], y[i-1])
    return t, y

# ── ODE — Runge-Kutta 4 ───────────────────────────────────────
def runge_kutta4(f, double y0, double t0, double tf, double dt=0.01):
    cdef int n = int((tf - t0) / dt) + 1
    cdef np.ndarray[double] t = np.linspace(t0, tf, n)
    cdef np.ndarray[double] y = np.zeros(n, dtype=np.float64)
    cdef double k1, k2, k3, k4
    cdef int i

    y[0] = y0
    for i in range(1, n):
        k1 = f(t[i-1],            y[i-1])
        k2 = f(t[i-1] + dt/2.0,   y[i-1] + dt/2.0 * k1)
        k3 = f(t[i-1] + dt/2.0,   y[i-1] + dt/2.0 * k2)
        k4 = f(t[i-1] + dt,        y[i-1] + dt * k3)
        y[i] = y[i-1] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return t, y

# ── Racines — Newton-Raphson ──────────────────────────────────
def newton(f, double x0, double tol=1e-8, int max_iter=100):
    cdef double x = x0
    cdef double fx, dfx, h
    cdef int i

    h = 1e-5
    for i in range(max_iter):
        fx  = f(x)
        dfx = (f(x + h) - f(x - h)) / (2.0 * h)
        if fabs(dfx) < 1e-14:
            raise ValueError("Dérivée nulle — Newton ne converge pas")
        x = x - fx / dfx
        if fabs(f(x)) < tol:
            return x
    raise ValueError(f"Newton n'a pas convergé en {max_iter} itérations")

# ── Racines — Bissection ──────────────────────────────────────
def bisection(f, double a, double b, double tol=1e-8, int max_iter=100):
    cdef double fa = f(a)
    cdef double fb = f(b)
    cdef double mid, fmid
    cdef int i

    if fa * fb > 0:
        raise ValueError("f(a) et f(b) doivent être de signes opposés")

    for i in range(max_iter):
        mid  = (a + b) / 2.0
        fmid = f(mid)
        if fabs(fmid) < tol:
            return mid
        if fa * fmid < 0:
            b = mid
        else:
            a = mid
            fa = fmid
    return (a + b) / 2.0