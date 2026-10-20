// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// TransFuser's fusion residual, x + bilinear_upsample(grid) with align_corners=False, in one pass over channels-last
// bf16 maps with fp32 arithmetic; the backward sums the gradient back onto the grid in two separable passes (width,
// then height) through an fp32 buffer. The source index follows PyTorch's upsample_bilinear2d: src = scale (dst + 0.5)
// - 0.5 clamped below at 0, taps floor(src) and floor(src) + 1 (the last row or column repeats).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include <cstdint>

namespace py123d_garage_cuda {
namespace {

// Channels per thread: one 16-byte vector of bf16.
constexpr int kVector = 8;
constexpr int kThreads = 256;

struct Taps {
  int low, high;
  float high_weight;
};

__device__ __forceinline__ Taps source_taps(int dst, float scale, int in_size) {
  const float src = fmaxf(scale * (static_cast<float>(dst) + 0.5f) - 0.5f, 0.0f);
  const int low = static_cast<int>(src);
  return {low, low + (low < in_size - 1 ? 1 : 0), src - static_cast<float>(low)};
}

// The weight of input index i in output dst's interpolation.
__device__ __forceinline__ float tap_weight(const Taps& taps, int i) {
  return (taps.low == i ? 1.0f - taps.high_weight : 0.0f) + (taps.high == i ? taps.high_weight : 0.0f);
}

__device__ __forceinline__ void load_vector(const __nv_bfloat16* pointer, float (&values)[kVector]) {
  const uint4 raw = *reinterpret_cast<const uint4*>(pointer);
  const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int k = 0; k < kVector / 2; ++k) {
    const float2 pair = __bfloat1622float2(pairs[k]);
    values[2 * k] = pair.x;
    values[2 * k + 1] = pair.y;
  }
}

__device__ __forceinline__ void store_vector(__nv_bfloat16* pointer, const float (&values)[kVector]) {
  uint4 raw;
  __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
  for (int k = 0; k < kVector / 2; ++k) pairs[k] = __floats2bfloat162_rn(values[2 * k], values[2 * k + 1]);
  *reinterpret_cast<uint4*>(pointer) = raw;
}

// out[b, y, x, :] = x[b, y, x, :] + bilinear grid value, all [B, rows, cols, C] channels-last.
__global__ void upsample_add_forward_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ grid,
                                            __nv_bfloat16* __restrict__ out, int64_t total, int channels, int height,
                                            int width, int grid_height, int grid_width, float scale_y, float scale_x) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int groups = channels / kVector;
  const int group = static_cast<int>(index % groups);
  int64_t pixel = index / groups;
  const int col = static_cast<int>(pixel % width);
  pixel /= width;
  const int row = static_cast<int>(pixel % height);
  const int64_t batch = pixel / height;
  const Taps ty = source_taps(row, scale_y, grid_height);
  const Taps tx = source_taps(col, scale_x, grid_width);
  const __nv_bfloat16* grid_batch = grid + batch * grid_height * grid_width * channels + group * kVector;
  const int64_t offsets[4] = {
      (static_cast<int64_t>(ty.low) * grid_width + tx.low) * channels,
      (static_cast<int64_t>(ty.low) * grid_width + tx.high) * channels,
      (static_cast<int64_t>(ty.high) * grid_width + tx.low) * channels,
      (static_cast<int64_t>(ty.high) * grid_width + tx.high) * channels,
  };
  const float weights[4] = {
      (1.0f - ty.high_weight) * (1.0f - tx.high_weight),
      (1.0f - ty.high_weight) * tx.high_weight,
      ty.high_weight * (1.0f - tx.high_weight),
      ty.high_weight * tx.high_weight,
  };
  float sum[kVector];
  load_vector(x + index * kVector, sum);
#pragma unroll
  for (int tap = 0; tap < 4; ++tap) {
    float values[kVector];
    load_vector(grid_batch + offsets[tap], values);
#pragma unroll
    for (int k = 0; k < kVector; ++k) sum[k] = fmaf(weights[tap], values[k], sum[k]);
  }
  store_vector(out + index * kVector, sum);
}

// partial[b, y, j, :] = sum over x of weight(x -> j) grad[b, y, x, :]; partial is fp32 [B, height, grid_width, C].
__global__ void upsample_backward_width_kernel(const __nv_bfloat16* __restrict__ grad, float* __restrict__ partial,
                                               int64_t total, int channels, int width, int grid_width, float scale_x) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int groups = channels / kVector;
  const int group = static_cast<int>(index % groups);
  const int64_t pixel = index / groups;
  const int j = static_cast<int>(pixel % grid_width);
  const int64_t row = pixel / grid_width;
  const float inverse = 1.0f / scale_x;
  const int first = max(0, static_cast<int>(floorf((j - 0.5f) * inverse - 0.5f)));
  const int last =
      j == grid_width - 1 ? width - 1 : min(width - 1, static_cast<int>(ceilf((j + 1.5f) * inverse - 0.5f)));
  float sum[kVector] = {};
  const __nv_bfloat16* grad_row = grad + row * width * channels + group * kVector;
  for (int col = first; col <= last; ++col) {
    const float weight = tap_weight(source_taps(col, scale_x, grid_width), j);
    if (weight == 0.0f) continue;
    float values[kVector];
    load_vector(grad_row + static_cast<int64_t>(col) * channels, values);
#pragma unroll
    for (int k = 0; k < kVector; ++k) sum[k] = fmaf(weight, values[k], sum[k]);
  }
  float4* target = reinterpret_cast<float4*>(partial + index * kVector);
  target[0] = make_float4(sum[0], sum[1], sum[2], sum[3]);
  target[1] = make_float4(sum[4], sum[5], sum[6], sum[7]);
}

// grad_grid[b, i, j, :] = sum over y of weight(y -> i) partial[b, y, j, :], bf16 [B, grid_height, grid_width, C].
__global__ void upsample_backward_height_kernel(const float* __restrict__ partial,
                                                __nv_bfloat16* __restrict__ grad_grid, int64_t total, int channels,
                                                int height, int grid_height, int grid_width, float scale_y) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int groups = channels / kVector;
  const int group = static_cast<int>(index % groups);
  int64_t pixel = index / groups;
  const int j = static_cast<int>(pixel % grid_width);
  pixel /= grid_width;
  const int i = static_cast<int>(pixel % grid_height);
  const int64_t batch = pixel / grid_height;
  const float inverse = 1.0f / scale_y;
  const int first = max(0, static_cast<int>(floorf((i - 0.5f) * inverse - 0.5f)));
  const int last =
      i == grid_height - 1 ? height - 1 : min(height - 1, static_cast<int>(ceilf((i + 1.5f) * inverse - 0.5f)));
  float sum[kVector] = {};
  const float* column = partial + ((batch * height) * grid_width + j) * channels + group * kVector;
  for (int row = first; row <= last; ++row) {
    const float weight = tap_weight(source_taps(row, scale_y, grid_height), i);
    if (weight == 0.0f) continue;
    const float4* source = reinterpret_cast<const float4*>(column + static_cast<int64_t>(row) * grid_width * channels);
    const float4 low = source[0], high = source[1];
    const float values[kVector] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
#pragma unroll
    for (int k = 0; k < kVector; ++k) sum[k] = fmaf(weight, values[k], sum[k]);
  }
  store_vector(grad_grid + index * kVector, sum);
}

int blocks_for(int64_t total) { return static_cast<int>((total + kThreads - 1) / kThreads); }

void check_channels_last(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.dim() == 4 && tensor.scalar_type() == at::kBFloat16,
              "fusion_upsample_add: ", name, " must be a 4D bf16 CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(at::MemoryFormat::ChannelsLast), "fusion_upsample_add: ", name,
              " must be channels-last");
  TORCH_CHECK(tensor.size(1) % kVector == 0, "fusion_upsample_add: channels must be a multiple of 8");
}

at::Tensor fusion_upsample_add(const at::Tensor& x, const at::Tensor& grid) {
  check_channels_last(x, "x");
  check_channels_last(grid, "grid");
  TORCH_CHECK(x.size(0) == grid.size(0) && x.size(1) == grid.size(1), "fusion_upsample_add: batch or channels differ");
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty_like(x, x.options(), at::MemoryFormat::ChannelsLast);
  const int channels = static_cast<int>(x.size(1));
  const int height = static_cast<int>(x.size(2)), width = static_cast<int>(x.size(3));
  const int grid_height = static_cast<int>(grid.size(2)), grid_width = static_cast<int>(grid.size(3));
  const int64_t total = x.numel() / kVector;
  if (total == 0) return out;
  upsample_add_forward_kernel<<<blocks_for(total), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(grid.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), total, channels, height, width, grid_height, grid_width,
      static_cast<float>(grid_height) / height, static_cast<float>(grid_width) / width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor fusion_upsample_add_backward(const at::Tensor& grad, int64_t grid_height, int64_t grid_width) {
  check_channels_last(grad, "grad");
  const c10::cuda::CUDAGuard guard(grad.device());
  const int64_t batch = grad.size(0);
  const int channels = static_cast<int>(grad.size(1));
  const int height = static_cast<int>(grad.size(2)), width = static_cast<int>(grad.size(3));
  auto partial = at::empty({batch, height, grid_width, channels}, grad.options().dtype(at::kFloat));
  auto grad_grid =
      at::empty({batch, channels, grid_height, grid_width}, grad.options(), at::MemoryFormat::ChannelsLast);
  const int64_t partial_total = partial.numel() / kVector;
  const int64_t grid_total = grad_grid.numel() / kVector;
  if (grid_total == 0) return grad_grid;
  const auto stream = at::cuda::getCurrentCUDAStream();
  upsample_backward_width_kernel<<<blocks_for(partial_total), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr()), partial.data_ptr<float>(), partial_total, channels,
      width, static_cast<int>(grid_width), static_cast<float>(grid_width) / width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  upsample_backward_height_kernel<<<blocks_for(grid_total), kThreads, 0, stream>>>(
      partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(grad_grid.data_ptr()), grid_total, channels, height,
      static_cast<int>(grid_height), static_cast<int>(grid_width), static_cast<float>(grid_height) / height);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_grid;
}

}  // namespace

TORCH_LIBRARY_IMPL(py123d_garage_cuda, CUDA, m) {
  m.impl("fusion_upsample_add", &fusion_upsample_add);
  m.impl("fusion_upsample_add_backward", &fusion_upsample_add_backward);
}

}  // namespace py123d_garage_cuda
