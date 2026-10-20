// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// The tensor-core depthwise 7 x 7 kernels of depthwise_conv_tc.cu.

#pragma once

#include <ATen/ATen.h>

#include <tuple>

namespace py123d_garage_cuda {
namespace depthwise_tc {

// Whether the tensor-core kernels take x: bf16 channels-last with C % 16 == 0, a bf16 7 x 7 kernel, sm_90 or newer.
bool supported(const at::Tensor& x, const at::Tensor& weight);

// conv2d(x, weight, bias, padding=3, groups=C), or with flip the same with the kernel rotated by 180 degrees.
at::Tensor conv(const at::Tensor& x, const at::Tensor& weight, const at::Tensor* bias, bool flip);

// The weight and bias gradients for the output gradient grad, in weight's dtype.
std::tuple<at::Tensor, at::Tensor> weight_grad(const at::Tensor& x, const at::Tensor& grad, const at::Tensor& weight);

}  // namespace depthwise_tc
}  // namespace py123d_garage_cuda
