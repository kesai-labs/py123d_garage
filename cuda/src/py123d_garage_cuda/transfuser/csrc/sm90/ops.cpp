// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// The op schemas of the kernels built on sm_90 instructions, TMA, mbarrier and bulk copies, which newer GPUs
// also run.

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(py123d_garage_cuda, m) {
  m.def("resnet_stem_conv(Tensor x, Tensor weight) -> (Tensor, Tensor)");
  m.def("resnet_stem_conv_weight_grad(Tensor grad, Tensor x, int channels) -> Tensor");
  m.def("batch_norm_finalize(Tensor partial, Tensor weight, Tensor bias, float eps) -> Tensor");
  m.def("regnet_grouped_conv(Tensor x, Tensor weight, int stride) -> Tensor");
  m.def("regnet_grouped_conv_backward(Tensor grad, Tensor x, Tensor weight, int stride) -> (Tensor, Tensor)");
  m.def(
      "regnet_bn_relu_grouped_conv(Tensor y, Tensor scale, Tensor shift, Tensor center, Tensor weight, int stride) -> "
      "(Tensor, Tensor)");
  m.def(
      "regnet_bn_relu_grouped_conv_backward(Tensor grad, Tensor y, Tensor scale, Tensor shift, Tensor center, "
      "Tensor weight, int stride) -> (Tensor, Tensor, Tensor)");
}
