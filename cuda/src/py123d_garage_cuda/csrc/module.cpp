// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// The py123d_garage_cuda._C extension module. Importing it loads the library, which registers the ops of
// torch.ops.py123d_garage_cuda: the schemas in transfuser/<gpu architecture>/csrc/ops.cpp, the CUDA implementations
// next to them, and the fake implementations and autograd formulas in the Python modules beside the kernels.

#include <Python.h>

extern "C" {
// The module itself is empty.
PyObject* PyInit__C(void) {  // NOLINT(bugprone-reserved-identifier)
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr, nullptr, nullptr, nullptr, nullptr};
  return PyModule_Create(&module_def);
}
}
