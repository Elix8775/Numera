# Numera

A fast scientific math library for Python, powered by Cython.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyPI](https://img.shields.io/pypi/v/Numera)
![License](https://img.shields.io/badge/license-MIT-green)

## Installation
```bash
pip install Numera
```

## Modules

| Module | Description |
|---|---|
| `Numera.Core.Matrices` | Linear algebra — dot product, inverse, determinant, eigenvalues |
| `Numera.Stats` | Descriptive statistics — mean, std, skewness, correlation... |
| `Numera.Calculus` | Numerical methods — derivatives, integrals, ODE, root finding |
| `Numera.Math` | Math functions — log, exp, sqrt, factorial, combinations |

## Quick start

### Matrices
```python
from Numera.Core.Matrices import Matrix

A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print(A @ B)          # matrix product → [[19, 22], [43, 50]]
print(A.det())        # determinant    → -2.0
print(A.inv())        # inverse
print(A.T)            # transpose
vals, vecs = A.eigenvalues()
```

### Statistics
```python
from Numera.Stats import Stats

s = Stats([4, 7, 13, 2, 1, 9, 15, 3])

print(s.mean())       # 6.75
print(s.median())     # 5.5
print(s.std())        # 4.86
print(s.skewness())   # asymmetry
print(s.percentile(75))

print(Stats.correlation([1, 2, 3, 4], [2, 4, 6, 8]))  # 1.0
```

### Calculus
```python
from Numera.Calculus import Calculus
import math

# Derivative
Calculus.derivative(math.sin, 0)           # ≈ 1.0

# Integral
Calculus.integrate_simpson(lambda x: x**2, 0, 1)  # ≈ 0.333

# ODE solver — dy/dt = -y
t, y = Calculus.runge_kutta4(lambda t, y: -y, y0=1.0, t0=0, tf=5)

# Root finding — √2
Calculus.newton(lambda x: x**2 - 2, x0=1.0)  # ≈ 1.4142
```

### Math
```python
from Numera.Math import Math

Math.ln(2.718)           # ≈ 1.0
Math.log(1000)           # 3.0
Math.factorial(10)       # 3628800
Math.C(10, 3)            # 120  — combinations
Math.P(5, 2)             # 20   — permutations
Math.cumsum([1, 2, 3, 4])  # [1, 3, 6, 10]
Math.exp([0, 1, 2])        # [1.0, 2.718, 7.389]
```

## Performance

Numera uses Cython extensions compiled to native C with:
- No bounds checking
- No wraparound
- Fast C division
- `-O3` compiler optimization

## Requirements

- Python 3.9+
- NumPy >= 1.21

## License

MIT# Numera
