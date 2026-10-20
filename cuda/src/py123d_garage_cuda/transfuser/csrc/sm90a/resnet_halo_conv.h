// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// The halo convolution of resnet_halo_conv.cu, for the pinned convolution ops.

#pragma once

#include <ATen/ATen.h>

namespace py123d_garage_cuda {

// Whether halo_conv runs this convolution: 3 x 3, stride 1, padding 1, 64 to 64 channels, on an sm_90 GPU.
bool halo_conv_supports(const at::Tensor& x, const at::Tensor& weight, at::IntArrayRef stride, at::IntArrayRef padding);

// conv2d(x, weight, padding=1), or with data_grad the input gradient of that convolution for the output gradient x;
// channels-last bf16 tensors, fp32 accumulation, a channels-last [64, 64, 3, 3] bf16 weight.
at::Tensor halo_conv(const at::Tensor& x, const at::Tensor& weight, bool data_grad);

// The weight gradient of conv2d(x, weight, padding=1) for the output gradient grad, as a channels-last [64, 64, 3, 3]
// bf16 tensor; partial sums per block in fp32, added in a fixed order.
at::Tensor halo_conv_weight_grad(const at::Tensor& grad, const at::Tensor& x);

}  // namespace py123d_garage_cuda
