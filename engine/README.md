# Uni-Dock2 C++ engine

This directory contains the private C++/CUDA implementation used by the
`unidock2` Python package. It does not provide a standalone product interface.

The Python binding in `api/python` is the only runtime adapter. It converts
Python inputs into `CoreInput` and then calls `core_pipeline()`.

## Developer build

Requirements:

- CUDA 11.8+
- CMake 3.27+
- Python 3.10+
- pybind11

```sh
cmake -S engine -B build/engine \
  -DBUILD_API=ON \
  -DBUILD_TEST=ON
cmake --build build/engine
ctest --test-dir build/engine --output-on-failure
```

The test suite contains C++ unit tests and pybind integration tests. The
integration tests load prepared engine JSON fixtures, execute docking through
the Python binding, and validate the generated poses.

## Native debugging through Python

Build the engine with debug symbols, then configure the debugger to launch the
Python interpreter with a focused pytest case. For example, use the equivalent
of:

```sh
PYTHONPATH=build/engine/api/python \
  python -m pytest engine/api/python/test/test_feature_pipeline.py \
  -k test_5s8i_best_pose_rmsd
```

Set breakpoints in `api/python/pipeline.cpp`, `screening/core.cpp`, or the
relevant C++/CUDA implementation. The debugger enters native code when the
Python test calls the binding.
