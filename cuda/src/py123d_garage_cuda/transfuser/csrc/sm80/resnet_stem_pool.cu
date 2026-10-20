// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// The tail of ResNet's stem in training, max_pool2d(relu(batch_norm(y)), 3, 2, 1) with the batch statistics given, on
// channels-last bf16 tensors whose channel count is a multiple of 8.
//
// The backward walks virtual windows (oh, ow) of the (OH + 1) x (OW + 1) grid, each owning the input pixels
// (2 oh - 1 | 2 oh, 2 ow - 1 | 2 ow), which receive gradient only from windows (oh - 1 | oh, ow - 1 | ow).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include <cstdint>

namespace py123d_garage_cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kVector = 8;
constexpr int kReduceBlocksPerSm = 4;

struct PoolShape {
  int batch, height, width, channels, out_height, out_width;

  __device__ __forceinline__ int groups() const { return channels / kVector; }
};

__device__ __forceinline__ void unpack(const int4& raw, float (&v)[kVector]) {
  const uint32_t words[4] = {static_cast<uint32_t>(raw.x), static_cast<uint32_t>(raw.y), static_cast<uint32_t>(raw.z),
                             static_cast<uint32_t>(raw.w)};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[2 * i] = __uint_as_float(words[i] << 16);
    v[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
  }
}

__device__ __forceinline__ int4 pack(const float (&v)[kVector]) {
  uint32_t words[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    __nv_bfloat162 pair = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
    words[i] = *reinterpret_cast<uint32_t*>(&pair);
  }
  return int4{static_cast<int>(words[0]), static_cast<int>(words[1]), static_cast<int>(words[2]),
              static_cast<int>(words[3])};
}

__device__ __forceinline__ float round_bf16(float value) { return __bfloat162float(__float2bfloat16(value)); }

__device__ __forceinline__ void load_values(const float* values, int row, int channels, int c0, float (&v)[kVector]) {
  const float4 lo = *reinterpret_cast<const float4*>(values + row * channels + c0);
  const float4 hi = *reinterpret_cast<const float4*>(values + row * channels + c0 + 4);
  v[0] = lo.x, v[1] = lo.y, v[2] = lo.z, v[3] = lo.w, v[4] = hi.x, v[5] = hi.y, v[6] = hi.z, v[7] = hi.w;
}

// out = max_pool(relu(y * scale + shift)) with the normalized values rounded to bf16 before comparing, as eager torch
// does, and the int8 window position kh * 3 + kw of the first maximum; one thread per 8 channels of an output pixel.
__global__ void __launch_bounds__(kThreads)
    stem_pool_forward_kernel(const __nv_bfloat16* __restrict__ y, const float* __restrict__ values,
                             __nv_bfloat16* __restrict__ out, int8_t* __restrict__ offsets, PoolShape shape) {
  const int groups = shape.groups();
  // Index math in 32 bits: the host checks that y has fewer than 2^31 elements.
  const int items = shape.batch * shape.out_height * shape.out_width * groups;
  for (int item = blockIdx.x * kThreads + threadIdx.x; item < items; item += gridDim.x * kThreads) {
    const int group = item % groups;
    const int pixel = item / groups;
    const int ow = pixel % shape.out_width;
    const int oh = pixel / shape.out_width % shape.out_height;
    const int n = pixel / (shape.out_width * shape.out_height);
    const int c0 = group * kVector;
    float scale[kVector], shift[kVector], best[kVector];
    int position[kVector];
    load_values(values, 3, shape.channels, c0, scale);
    load_values(values, 4, shape.channels, c0, shift);
#pragma unroll
    for (int i = 0; i < kVector; ++i) {
      best[i] = -INFINITY;
      position[i] = 0;
    }
#pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
      const int ih = 2 * oh - 1 + kh;
#pragma unroll
      for (int kw = 0; kw < 3; ++kw) {
        const int iw = 2 * ow - 1 + kw;
        if (ih < 0 || ih >= shape.height || iw < 0 || iw >= shape.width) continue;
        const int4 raw = *reinterpret_cast<const int4*>(
            y + ((static_cast<int64_t>(n) * shape.height + ih) * shape.width + iw) * shape.channels + c0);
        float v[kVector];
        unpack(raw, v);
#pragma unroll
        for (int i = 0; i < kVector; ++i) {
          const float z = fmaxf(round_bf16(v[i] * scale[i] + shift[i]), 0.0f);
          if (z > best[i]) {
            best[i] = z;
            position[i] = kh * 3 + kw;
          }
        }
      }
    }
    const int64_t index = static_cast<int64_t>(pixel) * shape.channels + c0;
    *reinterpret_cast<int4*>(out + index) = pack(best);
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      lo |= static_cast<uint32_t>(position[i]) << (8 * i);
      hi |= static_cast<uint32_t>(position[i + 4]) << (8 * i);
    }
    *reinterpret_cast<uint2*>(offsets + index) = make_uint2(lo, hi);
  }
}

// The backward kernels take 4 channels a thread, which keeps them near 64 registers.
constexpr int kBackwardVector = 4;

__device__ __forceinline__ void unpack4(const uint2& raw, float (&v)[kBackwardVector]) {
  v[0] = __uint_as_float(raw.x << 16);
  v[1] = __uint_as_float(raw.x & 0xffff0000u);
  v[2] = __uint_as_float(raw.y << 16);
  v[3] = __uint_as_float(raw.y & 0xffff0000u);
}

__device__ __forceinline__ uint2 pack4(const float (&v)[kBackwardVector]) {
  __nv_bfloat162 lo = __floats2bfloat162_rn(v[0], v[1]);
  __nv_bfloat162 hi = __floats2bfloat162_rn(v[2], v[3]);
  return make_uint2(*reinterpret_cast<uint32_t*>(&lo), *reinterpret_cast<uint32_t*>(&hi));
}

__device__ __forceinline__ void load_values4(const float* values, int row, int channels, int c0,
                                             float (&v)[kBackwardVector]) {
  const float4 value = *reinterpret_cast<const float4*>(values + row * channels + c0);
  v[0] = value.x, v[1] = value.y, v[2] = value.z, v[3] = value.w;
}

// Calls consume(dz, y, ih, iw) for each input pixel inside the image that virtual window (oh, ow) owns, with its
// gradient dz and value y for 4 channels. Pixel (dh, dw) is input (2 oh - 1 + dh, 2 ow - 1 + dw); window
// (oh - 1 + wh, ow - 1 + ww) covers it when dh + 2 - 2 wh and dw + 2 - 2 ww are window positions, so pixels in an even
// row or column of the window grid hear from one window along that axis and the others from two.
template <typename Consume>
__device__ __forceinline__ void for_owned_pixels(const __nv_bfloat16* __restrict__ grad,
                                                 const __nv_bfloat16* __restrict__ y,
                                                 const int8_t* __restrict__ offsets,
                                                 const float (&scale)[kBackwardVector],
                                                 const float (&shift)[kBackwardVector], const PoolShape& shape, int n,
                                                 int oh, int ow, int c0, Consume consume) {
  // All loads first, as raw words, so that an item costs one memory round trip: the 4 windows' gradients and
  // positions (no match outside the output) and the 4 pixels' values.
  uint2 window_grad[4];
  uint32_t window_position[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    const int window_h = oh - 1 + w / 2;
    const int window_w = ow - 1 + w % 2;
    window_grad[w] = make_uint2(0, 0);
    window_position[w] = 0xffffffffu;
    if (window_h >= 0 && window_h < shape.out_height && window_w >= 0 && window_w < shape.out_width) {
      const int64_t index =
          ((static_cast<int64_t>(n) * shape.out_height + window_h) * shape.out_width + window_w) * shape.channels + c0;
      window_grad[w] = *reinterpret_cast<const uint2*>(grad + index);
      window_position[w] = *reinterpret_cast<const uint32_t*>(offsets + index);
    }
  }
  uint2 pixel_value[4];
  bool inside[4];
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const int ih = 2 * oh - 1 + p / 2;
    const int iw = 2 * ow - 1 + p % 2;
    inside[p] = ih >= 0 && ih < shape.height && iw >= 0 && iw < shape.width;
    pixel_value[p] =
        inside[p] ? *reinterpret_cast<const uint2*>(
                        y + ((static_cast<int64_t>(n) * shape.height + ih) * shape.width + iw) * shape.channels + c0)
                  : make_uint2(0, 0);
  }
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    if (!inside[p]) continue;
    const int dh = p / 2, dw = p % 2;
    float value[kBackwardVector];
    unpack4(pixel_value[p], value);
    float dz[kBackwardVector] = {};
#pragma unroll
    for (int wh = dh; wh < 2; ++wh) {
#pragma unroll
      for (int ww = dw; ww < 2; ++ww) {
        const int w = 2 * wh + ww;
        const uint32_t target = static_cast<uint32_t>((dh + 2 - 2 * wh) * 3 + (dw + 2 - 2 * ww)) * 0x01010101u;
        const uint32_t match = __vcmpeq4(window_position[w], target);
        float g[kBackwardVector];
        unpack4(window_grad[w], g);
#pragma unroll
        for (int i = 0; i < kBackwardVector; ++i) dz[i] += (match >> (8 * i)) & 1 ? g[i] : 0.0f;
      }
    }
    // dz rounds to bf16 like torch's max-pool backward; the ReLU passes it where the normalized value is positive.
#pragma unroll
    for (int i = 0; i < kBackwardVector; ++i) {
      const bool passes = round_bf16(value[i] * scale[i] + shift[i]) > 0.0f;
      dz[i] = passes ? round_bf16(dz[i]) : 0.0f;
    }
    consume(dz, value, 2 * oh - 1 + dh, 2 * ow - 1 + dw);
  }
}

// partial[block, 0 | 1, c]: the block's sums of dz and dz * x_hat, dz being the pooled gradient through the ReLU.
__global__ void __launch_bounds__(kThreads)
    stem_pool_backward_reduce_kernel(const __nv_bfloat16* __restrict__ grad, const __nv_bfloat16* __restrict__ y,
                                     const int8_t* __restrict__ offsets, const float* __restrict__ values,
                                     float* __restrict__ partial, PoolShape shape) {
  extern __shared__ float sums[];
  const int groups = shape.channels / kBackwardVector;
  // Block and grid sizes are multiples of the group count, so a thread keeps its channel group.
  const int group = threadIdx.x % groups;
  const int c0 = group * kBackwardVector;
  float scale[kBackwardVector], shift[kBackwardVector], mean[kBackwardVector], rstd[kBackwardVector];
  load_values4(values, 0, shape.channels, c0, mean);
  load_values4(values, 2, shape.channels, c0, rstd);
  load_values4(values, 3, shape.channels, c0, scale);
  load_values4(values, 4, shape.channels, c0, shift);
  float sum_dz[kBackwardVector] = {}, sum_dz_x_hat[kBackwardVector] = {};
  const int virtual_h = shape.out_height + 1;
  const int virtual_w = shape.out_width + 1;
  const int items = shape.batch * virtual_h * virtual_w * groups;
  for (int item = blockIdx.x * blockDim.x + threadIdx.x; item < items; item += gridDim.x * blockDim.x) {
    const int window = item / groups;
    const int ow = window % virtual_w;
    const int oh = window / virtual_w % virtual_h;
    const int n = window / (virtual_w * virtual_h);
    for_owned_pixels(grad, y, offsets, scale, shift, shape, n, oh, ow, c0,
                     [&](const float (&dz)[kBackwardVector], const float (&value)[kBackwardVector], int, int) {
#pragma unroll
                       for (int i = 0; i < kBackwardVector; ++i) {
                         sum_dz[i] += dz[i];
                         sum_dz_x_hat[i] = fmaf(dz[i], value[i] - mean[i], sum_dz_x_hat[i]);
                       }
                     });
  }
  // Per channel, the threads of its group in thread order.
#pragma unroll
  for (int i = 0; i < kBackwardVector; ++i) {
    sums[(2 * i) * blockDim.x + threadIdx.x] = sum_dz[i];
    sums[(2 * i + 1) * blockDim.x + threadIdx.x] = sum_dz_x_hat[i] * rstd[i];
  }
  __syncthreads();
  for (int c = threadIdx.x; c < shape.channels; c += blockDim.x) {
    const int i = c % kBackwardVector;
    float dz = 0.0f, dz_x_hat = 0.0f;
    for (int t = c / kBackwardVector; t < blockDim.x; t += groups) {
      dz += sums[(2 * i) * blockDim.x + t];
      dz_x_hat += sums[(2 * i + 1) * blockDim.x + t];
    }
    partial[(blockIdx.x * 2) * shape.channels + c] = dz;
    partial[(blockIdx.x * 2 + 1) * shape.channels + c] = dz_x_hat;
  }
}

// Adds the blocks' pieces of channel blockIdx.x in a fixed order into the weight and bias gradients and the
// coefficients[0 | 1 | 2, c] of dy = a dz + b y + c.
__global__ void __launch_bounds__(kThreads)
    stem_pool_backward_coefficients_kernel(const float* __restrict__ partial, const float* __restrict__ weight,
                                           const float* __restrict__ values, float* __restrict__ grad_weight,
                                           float* __restrict__ grad_bias, float* __restrict__ coefficients, int blocks,
                                           int channels, float count) {
  __shared__ float shared[2][kThreads];
  const int c = blockIdx.x;
  float dz = 0.0f, dz_x_hat = 0.0f;
  for (int b = threadIdx.x; b < blocks; b += kThreads) {
    dz += partial[(b * 2) * channels + c];
    dz_x_hat += partial[(b * 2 + 1) * channels + c];
  }
  shared[0][threadIdx.x] = dz;
  shared[1][threadIdx.x] = dz_x_hat;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared[0][threadIdx.x] += shared[0][threadIdx.x + stride];
      shared[1][threadIdx.x] += shared[1][threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float sum_dz = shared[0][0];
    const float sum_dz_x_hat = shared[1][0];
    const float rstd = values[2 * channels + c];
    const float a = weight[c] * rstd;
    const float b = -a * rstd * sum_dz_x_hat / count;
    grad_weight[c] = sum_dz_x_hat;
    grad_bias[c] = sum_dz;
    coefficients[c] = a;
    coefficients[channels + c] = b;
    coefficients[2 * channels + c] = -a * sum_dz / count - b * values[c];
  }
}

// dy = a dz + b y + c per channel with the coefficients of stem_pool_backward_coefficients_kernel.
__global__ void __launch_bounds__(kThreads)
    stem_pool_backward_apply_kernel(const __nv_bfloat16* __restrict__ grad, const __nv_bfloat16* __restrict__ y,
                                    const int8_t* __restrict__ offsets, const float* __restrict__ values,
                                    const float* __restrict__ coefficients, __nv_bfloat16* __restrict__ dy,
                                    PoolShape shape) {
  const int groups = shape.channels / kBackwardVector;
  // The block size and the grid stride are multiples of the group count, so a thread keeps its channel group.
  const int c0 = threadIdx.x % groups * kBackwardVector;
  float scale[kBackwardVector], shift[kBackwardVector], a[kBackwardVector], b[kBackwardVector], c[kBackwardVector];
  load_values4(values, 3, shape.channels, c0, scale);
  load_values4(values, 4, shape.channels, c0, shift);
  load_values4(coefficients, 0, shape.channels, c0, a);
  load_values4(coefficients, 1, shape.channels, c0, b);
  load_values4(coefficients, 2, shape.channels, c0, c);
  const int virtual_h = shape.out_height + 1;
  const int virtual_w = shape.out_width + 1;
  const int items = shape.batch * virtual_h * virtual_w * groups;
  for (int item = blockIdx.x * blockDim.x + threadIdx.x; item < items; item += gridDim.x * blockDim.x) {
    const int window = item / groups;
    const int ow = window % virtual_w;
    const int oh = window / virtual_w % virtual_h;
    const int n = window / (virtual_w * virtual_h);
    for_owned_pixels(
        grad, y, offsets, scale, shift, shape, n, oh, ow, c0,
        [&](const float (&dz)[kBackwardVector], const float (&value)[kBackwardVector], int ih, int iw) {
          float v[kBackwardVector];
#pragma unroll
          for (int i = 0; i < kBackwardVector; ++i) v[i] = a[i] * dz[i] + b[i] * value[i] + c[i];
          *reinterpret_cast<uint2*>(
              dy + ((static_cast<int64_t>(n) * shape.height + ih) * shape.width + iw) * shape.channels + c0) = pack4(v);
        });
  }
}

PoolShape make_shape(const at::Tensor& y) {
  PoolShape shape;
  shape.batch = static_cast<int>(y.size(0));
  shape.channels = static_cast<int>(y.size(1));
  shape.height = static_cast<int>(y.size(2));
  shape.width = static_cast<int>(y.size(3));
  shape.out_height = (shape.height - 1) / 2 + 1;
  shape.out_width = (shape.width - 1) / 2 + 1;
  return shape;
}

void check_tensor(const at::Tensor& tensor, at::ScalarType dtype, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.dim() == 4, name, " must be a 4-D CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(tensor.is_contiguous(at::MemoryFormat::ChannelsLast), name, " must be channels-last");
  TORCH_CHECK(tensor.size(1) % kVector == 0, name, " must have a multiple of 8 channels");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

void check_values(const at::Tensor& values, int64_t channels) {
  TORCH_CHECK(values.is_cuda() && values.scalar_type() == at::kFloat && values.is_contiguous() && values.dim() == 2 &&
                  values.size(1) == channels,
              "values must be contiguous fp32 [5, C]");
}

int grid_blocks(int64_t items, int per_sm) {
  const int64_t limit = static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * per_sm;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(limit, (items + kThreads - 1) / kThreads)));
}

// Forward op: the channels-last pooled output and int8 window positions of y [N, C, H, W] and values [5, C]. weight and
// bias are folded into values; the op takes them for its autograd formula, which returns their gradients.
std::tuple<at::Tensor, at::Tensor> resnet_stem_pool(const at::Tensor& y, const at::Tensor& values,
                                                    const at::Tensor& /*weight*/, const at::Tensor& /*bias*/) {
  check_tensor(y, at::kBFloat16, "resnet_stem_pool: y");
  TORCH_CHECK(y.numel() < (int64_t{1} << 31), "resnet_stem_pool: y is too large");
  check_values(values, y.size(1));
  const c10::cuda::CUDAGuard guard(y.device());
  const PoolShape shape = make_shape(y);
  const auto layout = y.options().memory_format(at::MemoryFormat::ChannelsLast);
  auto out = at::empty({shape.batch, shape.channels, shape.out_height, shape.out_width}, layout);
  auto offsets = at::empty({shape.batch, shape.channels, shape.out_height, shape.out_width}, layout.dtype(at::kChar));
  const int64_t items = out.numel() / kVector;
  stem_pool_forward_kernel<<<grid_blocks(items, 16), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), values.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), offsets.data_ptr<int8_t>(), shape);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, offsets};
}

// Backward op: the gradients of y (channels-last bf16), weight and bias, through the batch statistics too.
std::tuple<at::Tensor, at::Tensor, at::Tensor> resnet_stem_pool_backward(const at::Tensor& grad, const at::Tensor& y,
                                                                         const at::Tensor& offsets,
                                                                         const at::Tensor& values,
                                                                         const at::Tensor& weight) {
  check_tensor(y, at::kBFloat16, "resnet_stem_pool_backward: y");
  TORCH_CHECK(y.numel() < (int64_t{1} << 31), "resnet_stem_pool_backward: y is too large");
  const auto g = grad.contiguous(at::MemoryFormat::ChannelsLast);
  check_tensor(g, at::kBFloat16, "resnet_stem_pool_backward: grad");
  check_tensor(offsets, at::kChar, "resnet_stem_pool_backward: offsets");
  check_values(values, y.size(1));
  const c10::cuda::CUDAGuard guard(y.device());
  const PoolShape shape = make_shape(y);
  TORCH_CHECK(g.size(0) == shape.batch && g.size(1) == shape.channels && g.size(2) == shape.out_height &&
                  g.size(3) == shape.out_width && offsets.sizes() == g.sizes(),
              "resnet_stem_pool_backward: grad and offsets must have the pooled shape");
  const int groups = shape.channels / kBackwardVector;
  const int64_t items = static_cast<int64_t>(shape.batch) * (shape.out_height + 1) * (shape.out_width + 1) * groups;
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto f32 = y.options().dtype(at::kFloat);

  // Block size: a multiple of the channel groups, so that a thread keeps its group across the grid-stride loop.
  const int reduce_threads = kThreads / groups * groups;
  const int reduce_blocks = static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>(
             static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * kReduceBlocksPerSm,
             (items + reduce_threads - 1) / reduce_threads)));
  auto partial = at::empty({reduce_blocks, 2, shape.channels}, f32);
  stem_pool_backward_reduce_kernel<<<reduce_blocks, reduce_threads,
                                     2 * kBackwardVector * reduce_threads * sizeof(float), stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()),
      offsets.data_ptr<int8_t>(), values.data_ptr<float>(), partial.data_ptr<float>(), shape);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const auto w = weight.to(at::kFloat).contiguous();
  auto grad_weight = at::empty({shape.channels}, f32);
  auto grad_bias = at::empty({shape.channels}, f32);
  auto coefficients = at::empty({3, shape.channels}, f32);
  stem_pool_backward_coefficients_kernel<<<shape.channels, kThreads, 0, stream>>>(
      partial.data_ptr<float>(), w.data_ptr<float>(), values.data_ptr<float>(), grad_weight.data_ptr<float>(),
      grad_bias.data_ptr<float>(), coefficients.data_ptr<float>(), reduce_blocks, shape.channels,
      static_cast<float>(static_cast<int64_t>(shape.batch) * shape.height * shape.width));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto dy = at::empty_like(y, y.options().memory_format(at::MemoryFormat::ChannelsLast));
  stem_pool_backward_apply_kernel<<<grid_blocks(items, 16), reduce_threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()),
      offsets.data_ptr<int8_t>(), values.data_ptr<float>(), coefficients.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(dy.data_ptr()), shape);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dy, grad_weight.to(weight.scalar_type()), grad_bias.to(weight.scalar_type())};
}

}  // namespace

TORCH_LIBRARY_IMPL(py123d_garage_cuda, CUDA, m) {
  m.impl("resnet_stem_pool", &resnet_stem_pool);
  m.impl("resnet_stem_pool_backward", &resnet_stem_pool_backward);
}

}  // namespace py123d_garage_cuda
