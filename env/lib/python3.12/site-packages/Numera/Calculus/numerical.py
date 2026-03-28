import numpy as np
from . import _calculus_core as _core

class Calculus:

    @staticmethod
    def derivative(f, x, h=1e-5):
        """Dérivée en un point (différences finies centrées)"""
        return _core.derivative(f, x, h)

    @staticmethod
    def derivative_n(f, x, n=2, h=1e-5):
        """Dérivée d'ordre n en un point"""
        return _core.derivative_n(f, x, n, h)

    @staticmethod
    def gradient(f, point, h=1e-5):
        """Gradient d'une fonction multivariable ∇f(x, y, ...)"""
        point = np.array(point, dtype=np.float64)
        return _core.gradient(f, point, h)

    @staticmethod
    def integrate_trapeze(f, a, b, n=1000):
        """Intégrale par méthode des trapèzes"""
        return _core.integrate_trapeze(f, a, b, n)

    @staticmethod
    def integrate_simpson(f, a, b, n=1000):
        """Intégrale par méthode de Simpson (plus précise)"""
        if n % 2 != 0:
            n += 1 
        return _core.integrate_simpson(f, a, b, n)

    @staticmethod
    def integrate_romberg(f, a, b, tol=1e-8):
        """Intégrale par méthode de Romberg (très précise)"""
        return _core.integrate_romberg(f, a, b, tol)

    @staticmethod
    def euler(f, y0, t0, tf, dt=0.01):
        """Méthode d'Euler explicite  dy/dt = f(t, y)"""
        return _core.euler(f, y0, t0, tf, dt)

    @staticmethod
    def runge_kutta4(f, y0, t0, tf, dt=0.01):
        """Méthode Runge-Kutta ordre 4 (RK4)  dy/dt = f(t, y)"""
        return _core.runge_kutta4(f, y0, t0, tf, dt)

    @staticmethod
    def newton(f, x0, tol=1e-8, max_iter=100):
        """Méthode de Newton-Raphson pour trouver f(x) = 0"""
        return _core.newton(f, x0, tol, max_iter)

    @staticmethod
    def bisection(f, a, b, tol=1e-8, max_iter=100):
        """Méthode de bissection pour trouver f(x) = 0"""
        return _core.bisection(f, a, b, tol, max_iter)