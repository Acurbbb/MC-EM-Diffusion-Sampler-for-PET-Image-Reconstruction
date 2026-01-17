# 1) Simple build
# rm -rf build/ rayprj/*.so rayprj*.so
# python setup.py build_ext --inplace

# 2) Package build
# rm -rf build/ dist/ *.egg-info rayprj/*.so rayprj/*.pyd
# export PATH="/home/bybhuang/GCC-6.3.0/bin:$PATH"
# pip install -e .

# setup.py
from setuptools import setup, Extension
import pybind11

common_compile = ["-O3", "-DNDEBUG", "-fopenmp", "-DUSE_OMP", "-std=c++14"]
common_link    = ["-fopenmp"]

exts = []

# non-TOF: rayprj (fproj_mt, bproj_mt)
exts.append(Extension(
    "rayprj._core", # put in package (folder) rayprj
    sources=["./src/rayprj_py.cpp"],
    include_dirs=[pybind11.get_include(), "."],
    language="c++",
    extra_compile_args=common_compile,       # no -DUSE_TOF here
    extra_link_args=common_link
))

# TOF: rayprj_tof (fproj_tof_mt, bproj_tof_mt)
exts.append(Extension(
    "rayprj._core_tof",
    sources=["./src/rayprj_py.cpp"],
    include_dirs=[pybind11.get_include(), "."],
    define_macros=[("USE_TOF", "1")],
    language="c++",
    extra_compile_args=common_compile + ["-DUSE_TOF"],
    extra_link_args=common_link
))

setup(
    name="rayprj",
    version="0.1.0",
    description="PET ray projector/backprojector with pybind11 + OpenMP",
    packages=["rayprj"],           # <-- install the package directory
    ext_modules=exts,
    zip_safe=False,                # C-extensions are not zip-safe
)
