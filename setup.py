from setuptools import setup, find_packages
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np

extensions = [
    Extension(
        name="Numera.Core.Matrices._matrix_core",
        sources=["Numera/Core/Matrices/_matrix_core.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
        name="Numera.Stats._stats_core",
        sources=["Numera/Stats/_stats_core.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
    name="Numera.Calculus._calculus_core",
    sources=["Numera/Calculus/_calculus_core.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
    name="Numera.Math._math_core",
    sources=["Numera/Math/_math_core.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-march=native"],
    ),
]

compiler_directives = {
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "language_level": "3",
}

setup(
    name="Numera",
    version="0.1.1",
    description="A fast scientific math library powered by Cython",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Elix8",
    author_email="email@example.com",
    url="https://github.com/tonpseudo/Numera",
    license="MIT",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives=compiler_directives, annotate=True),
    install_requires=["numpy>=1.21"],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)