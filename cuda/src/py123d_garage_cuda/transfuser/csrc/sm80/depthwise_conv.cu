// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// ConvNeXt's depthwise k x k convolution (stride 1, padding k / 2) on channels-last tensors, accumulated in fp32.
// Dispatch, first match: the tensor-core kernels of depthwise_conv_tc.cu, the pipelined then the tiled shared-memory
// kernels for bf16, and one-output-per-thread kernels for every other case.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include "transfuser/csrc/sm80/ptx.cuh"
#include "transfuser/csrc/sm90/depthwise_conv_tc.h"

namespace py123d_garage_cuda {
namespace {

constexpr int kThreads = 256;
// Weight-gradient blocks of the fallback across all pixel chunks and channel blocks.
constexpr int kWeightGradBlocks = 2048;

// Tiled kernels: channel pairs per block, output tile rows, outputs per thread strip.
constexpr int kPairs = 16;
constexpr int kTileRows = 8;
constexpr int kStrip = 8;
// Tiled weight gradient: output tile rows, row groups per block, and blocks per channel chunk.
constexpr int kGradTileRows = 4;
constexpr int kGradRowGroups = 2;
constexpr int kGradTargetBlocks = 528;

// A shared-memory row stride, in 4-byte words, of at least `words` with stride % 32 == 16: two half-warps a row
// apart then hit disjoint banks.
__host__ __device__ constexpr int padded_stride(int words) { return words + ((16 - words % 32) + 32) % 32; }

__device__ __forceinline__ float2 load_pair(const __nv_bfloat162* __restrict__ p) { return __bfloat1622float2(*p); }

// One output per thread: y = conv2d(x, weight, bias) for channels-last x and a contiguous [C, 1, k, k] weight.
template <typename scalar_t>
__global__ void depthwise_conv_kernel(const scalar_t* __restrict__ x, const scalar_t* __restrict__ weight,
                                      const scalar_t* __restrict__ bias, scalar_t* __restrict__ y, int64_t total,
                                      int height, int width, int channels, int kernel) {
  const int64_t index = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int c = index % channels;
  int64_t pixel = index / channels;
  const int w = pixel % width;
  pixel /= width;
  const int h = pixel % height;
  const int64_t n = pixel / height;
  const int pad = kernel / 2;
  const scalar_t* weight_c = weight + static_cast<int64_t>(c) * kernel * kernel;
  float acc = bias != nullptr ? static_cast<float>(bias[c]) : 0.0f;
  for (int kh = 0; kh < kernel; ++kh) {
    const int ih = h + kh - pad;
    if (ih < 0 || ih >= height) {
      continue;
    }
    for (int kw = 0; kw < kernel; ++kw) {
      const int iw = w + kw - pad;
      if (iw < 0 || iw >= width) {
        continue;
      }
      acc += static_cast<float>(x[((n * height + ih) * width + iw) * channels + c]) *
             static_cast<float>(weight_c[kh * kernel + kw]);
    }
  }
  y[index] = static_cast<scalar_t>(acc);
}

// Per pixel chunk and channel: the partial weight gradient sum_p x[p + tap] * grad[p] into [chunks, C, k * k] and the
// partial bias gradient sum_p grad[p] into [chunks, C].
template <typename scalar_t, int KERNEL>
__global__ void depthwise_conv_weight_grad_kernel(const scalar_t* __restrict__ x, const scalar_t* __restrict__ grad,
                                                  float* __restrict__ weight_partial, float* __restrict__ bias_partial,
                                                  int64_t pixels, int64_t pixels_per_chunk, int height, int width,
                                                  int channels) {
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c >= channels) {
    return;
  }
  constexpr int pad = KERNEL / 2;
  float acc[KERNEL * KERNEL];
#pragma unroll
  for (int tap = 0; tap < KERNEL * KERNEL; ++tap) {
    acc[tap] = 0.0f;
  }
  float bias_acc = 0.0f;
  const int64_t begin = blockIdx.x * pixels_per_chunk;
  const int64_t end = min(begin + pixels_per_chunk, pixels);
  for (int64_t pixel = begin; pixel < end; ++pixel) {
    const float g = static_cast<float>(grad[pixel * channels + c]);
    bias_acc += g;
    const int w = pixel % width;
    const int h = (pixel / width) % height;
    const int64_t n = pixel / (static_cast<int64_t>(width) * height);
#pragma unroll
    for (int kh = 0; kh < KERNEL; ++kh) {
      const int ih = h + kh - pad;
      if (ih < 0 || ih >= height) {
        continue;
      }
#pragma unroll
      for (int kw = 0; kw < KERNEL; ++kw) {
        const int iw = w + kw - pad;
        if (iw < 0 || iw >= width) {
          continue;
        }
        acc[kh * KERNEL + kw] += static_cast<float>(x[((n * height + ih) * width + iw) * channels + c]) * g;
      }
    }
  }
  float* partial = weight_partial + (static_cast<int64_t>(blockIdx.x) * channels + c) * KERNEL * KERNEL;
#pragma unroll
  for (int tap = 0; tap < KERNEL * KERNEL; ++tap) {
    partial[tap] = acc[tap];
  }
  bias_partial[static_cast<int64_t>(blockIdx.x) * channels + c] = bias_acc;
}

// Stages the input tile rows [h0, h0 + ROWS) x columns [w0, w0 + COLS) of channel pairs [cp0, cp0 + kPairs) of image
// n into tile[row * STRIDE + col * kPairs + pair], zero outside the image and past the last pair.
template <int ROWS, int COLS, int STRIDE>
__device__ __forceinline__ void stage_tile(__nv_bfloat162* __restrict__ tile, const __nv_bfloat162* __restrict__ x,
                                           int64_t n, int h0, int w0, int cp0, int height, int width, int pairs) {
  for (int i = threadIdx.x; i < ROWS * COLS * kPairs; i += blockDim.x) {
    const int pair = i % kPairs;
    const int col = (i / kPairs) % COLS;
    const int row = i / (kPairs * COLS);
    const int ih = h0 + row;
    const int iw = w0 + col;
    const int cp = cp0 + pair;
    __nv_bfloat162 v = __floats2bfloat162_rn(0.0f, 0.0f);
    if (ih >= 0 && ih < height && iw >= 0 && iw < width && cp < pairs) {
      v = x[((n * height + ih) * width + iw) * pairs + cp];
    }
    tile[row * STRIDE + col * kPairs + pair] = v;
  }
}

// y = conv2d(x, weight, bias) for channels-last bf16 x and weight_t [k * k, C]; with FLIP the kernel is rotated by 180
// degrees (the input gradient). Threads: kPairs channel pairs x kTileRows rows x TW / kStrip strips; blocks:
// (image, tile row, tile column) x channel chunks.
template <int K, int TW, bool FLIP>
__global__ void __launch_bounds__(kPairs* kTileRows*(TW / kStrip))
    depthwise_conv_tiled_kernel(const __nv_bfloat162* __restrict__ x, const __nv_bfloat162* __restrict__ weight_t,
                                const __nv_bfloat162* __restrict__ bias, __nv_bfloat162* __restrict__ y, int height,
                                int width, int pairs) {
  constexpr int P = K / 2;
  constexpr int TR = kTileRows + K - 1;
  constexpr int TC = TW + K - 1;
  constexpr int STRIDE = padded_stride(TC * kPairs);
  __shared__ __nv_bfloat162 tile[TR * STRIDE];
  __shared__ float2 weights[K * K * kPairs];

  const int tiles_w = (width + TW - 1) / TW;
  const int tiles_h = (height + kTileRows - 1) / kTileRows;
  const int tile_w = blockIdx.x % tiles_w;
  const int tile_h = (blockIdx.x / tiles_w) % tiles_h;
  const int64_t n = blockIdx.x / (static_cast<int64_t>(tiles_w) * tiles_h);
  const int cp0 = blockIdx.y * kPairs;
  const int oh0 = tile_h * kTileRows;
  const int ow0 = tile_w * TW;

  for (int i = threadIdx.x; i < K * K * kPairs; i += blockDim.x) {
    const int pair = i % kPairs;
    const int tap = i / kPairs;
    const int source = FLIP ? K * K - 1 - tap : tap;
    weights[i] = cp0 + pair < pairs ? load_pair(weight_t + source * pairs + cp0 + pair) : make_float2(0.0f, 0.0f);
  }
  stage_tile<TR, TC, STRIDE>(tile, x, n, oh0 - P, ow0 - P, cp0, height, width, pairs);
  __syncthreads();

  const int pair = threadIdx.x % kPairs;
  const int row = (threadIdx.x / kPairs) % kTileRows;
  const int strip = threadIdx.x / (kPairs * kTileRows);
  const int cp = cp0 + pair;
  const int oh = oh0 + row;
  const int ow = ow0 + strip * kStrip;

  const float2 b = bias != nullptr && cp < pairs ? load_pair(bias + cp) : make_float2(0.0f, 0.0f);
  float2 acc[kStrip];
#pragma unroll
  for (int j = 0; j < kStrip; ++j) {
    acc[j] = b;
  }
#pragma unroll
  for (int kh = 0; kh < K; ++kh) {
    const __nv_bfloat162* source = tile + (row + kh) * STRIDE + strip * kStrip * kPairs + pair;
    float2 wk[K];
#pragma unroll
    for (int kw = 0; kw < K; ++kw) {
      wk[kw] = weights[(kh * K + kw) * kPairs + pair];
    }
#pragma unroll
    for (int j = 0; j < kStrip + K - 1; ++j) {
      const float2 v = __bfloat1622float2(source[j * kPairs]);
#pragma unroll
      for (int kw = 0; kw < K; ++kw) {
        const int o = j - kw;
        if (o >= 0 && o < kStrip) {
          acc[o].x = fmaf(v.x, wk[kw].x, acc[o].x);
          acc[o].y = fmaf(v.y, wk[kw].y, acc[o].y);
        }
      }
    }
  }
  if (cp >= pairs || oh >= height) {
    return;
  }
  __nv_bfloat162* out = y + ((n * height + oh) * width + ow) * pairs + cp;
#pragma unroll
  for (int j = 0; j < kStrip; ++j) {
    if (ow + j < width) {
      out[static_cast<int64_t>(j) * pairs] = __float22bfloat162_rn(acc[j]);
    }
  }
}

// Threads: kPairs channel pairs x K kernel rows x kGradRowGroups; blocks: slots x channel chunks, every slot sweeping
// a strided share of the (image, tile row, tile column) tiles. Each thread keeps the K taps of its kernel row of two
// channels; partials are [slot * kGradRowGroups + group][K * K][pairs] float2 and, from the kernel-row-0 threads,
// [slot * kGradRowGroups + group][pairs] float2 for the bias.
template <int K, int TW>
__global__ void __launch_bounds__(kPairs* K* kGradRowGroups)
    depthwise_conv_weight_grad_tiled_kernel(const __nv_bfloat162* __restrict__ x,
                                            const __nv_bfloat162* __restrict__ grad,
                                            float2* __restrict__ weight_partial, float2* __restrict__ bias_partial,
                                            int batch, int height, int width, int pairs) {
  constexpr int P = K / 2;
  constexpr int TR = kGradTileRows + K - 1;
  constexpr int TC = TW + K - 1;
  constexpr int STRIDE = padded_stride(TC * kPairs);
  constexpr int GRAD_STRIDE = padded_stride(TW * kPairs);
  __shared__ __nv_bfloat162 tile[TR * STRIDE];
  __shared__ __nv_bfloat162 grad_tile[kGradTileRows * GRAD_STRIDE];

  const int pair = threadIdx.x % kPairs;
  const int kh = (threadIdx.x / kPairs) % K;
  const int group = threadIdx.x / (kPairs * K);
  const int cp0 = blockIdx.y * kPairs;
  const int cp = cp0 + pair;
  const int tiles_w = (width + TW - 1) / TW;
  const int tiles_h = (height + kGradTileRows - 1) / kGradTileRows;
  const int64_t tiles = static_cast<int64_t>(batch) * tiles_h * tiles_w;

  float2 acc[K];
#pragma unroll
  for (int kw = 0; kw < K; ++kw) {
    acc[kw] = make_float2(0.0f, 0.0f);
  }
  float2 bias_acc = make_float2(0.0f, 0.0f);

  for (int64_t t = blockIdx.x; t < tiles; t += gridDim.x) {
    const int tile_w = t % tiles_w;
    const int tile_h = (t / tiles_w) % tiles_h;
    const int64_t n = t / (static_cast<int64_t>(tiles_w) * tiles_h);
    const int oh0 = tile_h * kGradTileRows;
    const int ow0 = tile_w * TW;
    __syncthreads();
    stage_tile<TR, TC, STRIDE>(tile, x, n, oh0 - P, ow0 - P, cp0, height, width, pairs);
    stage_tile<kGradTileRows, TW, GRAD_STRIDE>(grad_tile, grad, n, oh0, ow0, cp0, height, width, pairs);
    __syncthreads();
    for (int r = group; r < kGradTileRows; r += kGradRowGroups) {
      const __nv_bfloat162* source = tile + (r + kh) * STRIDE + pair;
      const __nv_bfloat162* g_row = grad_tile + r * GRAD_STRIDE + pair;
      float2 window[K];
#pragma unroll
      for (int q = 0; q < K - 1; ++q) {
        window[q] = __bfloat1622float2(source[q * kPairs]);
      }
#pragma unroll
      for (int j = 0; j < TW; ++j) {
        window[K - 1] = __bfloat1622float2(source[(j + K - 1) * kPairs]);
        const float2 g = __bfloat1622float2(g_row[j * kPairs]);
#pragma unroll
        for (int kw = 0; kw < K; ++kw) {
          acc[kw].x = fmaf(window[kw].x, g.x, acc[kw].x);
          acc[kw].y = fmaf(window[kw].y, g.y, acc[kw].y);
        }
        if (kh == 0) {
          bias_acc.x += g.x;
          bias_acc.y += g.y;
        }
#pragma unroll
        for (int q = 0; q < K - 1; ++q) {
          window[q] = window[q + 1];
        }
      }
    }
  }
  if (cp >= pairs) {
    return;
  }
  const int64_t slot = static_cast<int64_t>(blockIdx.x) * kGradRowGroups + group;
#pragma unroll
  for (int kw = 0; kw < K; ++kw) {
    weight_partial[(slot * K * K + kh * K + kw) * pairs + cp] = acc[kw];
  }
  if (kh == 0) {
    bias_partial[slot * pairs + cp] = bias_acc;
  }
}

// Pipelined kernels (C % 32 == 0): tile rows, output rows per thread, blocks per SM and strip width.
constexpr int kPipeTileRows = 8;
constexpr int kPipeRows = 2;
constexpr int kPipeBlocksPerSm = 3;
constexpr int kPipeStrip = 8;
__host__ __device__ constexpr int pipe_threads(int cols) {
  return kPairs * (kPipeTileRows / kPipeRows) * (cols / kPipeStrip);
}

// Row stride in words for the pipelined tiles: % 32 == 16 / kPipeRows, so half-warps kPipeRows rows apart use
// disjoint banks, and a multiple of 4 words, so every 16-byte cp.async destination is aligned.
__host__ __device__ constexpr int pipe_stride(int words) { return words + ((16 / kPipeRows - words % 32) + 32) % 32; }

template <int K, int TCOLS>
struct PipeShape {
  static constexpr int rows = kPipeTileRows + K - 1;
  static constexpr int cols = TCOLS + K - 1;
  static constexpr int stride = pipe_stride(cols * kPairs);
  static constexpr int buffer_words = rows * stride;
  static constexpr int bytes = 2 * buffer_words * 4 + K * K * kPairs * static_cast<int>(sizeof(float2));
};

// Issues the cp.async of one input tile: rows [h0, h0 + ROWS) x columns [w0, w0 + COLS) of the 16 channel pairs from
// cp0, zero-filled outside the image, into tile[row * STRIDE + col * kPairs + pair].
template <int ROWS, int COLS, int STRIDE>
__device__ __forceinline__ void issue_tile(__nv_bfloat162* tile, const __nv_bfloat162* __restrict__ x, int64_t n,
                                           int h0, int w0, int cp0, int height, int width, int pairs) {
  constexpr int chunks = kPairs / 4;
  for (int i = threadIdx.x; i < ROWS * COLS * chunks; i += blockDim.x) {
    const int chunk = i % chunks;
    const int col = (i / chunks) % COLS;
    const int row = i / (chunks * COLS);
    const int ih = h0 + row;
    const int iw = w0 + col;
    const bool valid = ih >= 0 && ih < height && iw >= 0 && iw < width;
    const __nv_bfloat162* src = valid ? x + ((n * height + ih) * width + iw) * pairs + cp0 + chunk * 4 : x;
    cp_async_16(tile + row * STRIDE + col * kPairs + chunk * 4, src, valid);
  }
}

// The tiled kernel's convolution with a contiguous [C, 1, k, k] weight: resident blocks loop over the tiles of one
// 32-channel chunk, prefetching the next tile by cp.async into the second of two shared buffers.
template <int K, int TCOLS, bool FLIP>
__global__ void __launch_bounds__(pipe_threads(TCOLS), kPipeBlocksPerSm)
    depthwise_conv_pipelined_kernel(const __nv_bfloat162* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
                                    const __nv_bfloat162* __restrict__ bias, __nv_bfloat162* __restrict__ y, int batch,
                                    int height, int width, int pairs) {
  using Shape = PipeShape<K, TCOLS>;
  constexpr int P = K / 2;
  extern __shared__ __align__(16) unsigned char shared[];
  auto* buffers = reinterpret_cast<__nv_bfloat162*>(shared);
  auto* weights = reinterpret_cast<float2*>(shared + 2 * Shape::buffer_words * 4);

  const int cp0 = blockIdx.y * kPairs;
  const int tiles_w = (width + TCOLS - 1) / TCOLS;
  const int tiles_h = (height + kPipeTileRows - 1) / kPipeTileRows;
  const int64_t tiles = static_cast<int64_t>(batch) * tiles_h * tiles_w;

  for (int i = threadIdx.x; i < K * K * kPairs; i += blockDim.x) {
    const int pair = i % kPairs;
    const int tap = i / kPairs;
    const int64_t source = static_cast<int64_t>(2 * (cp0 + pair)) * K * K + (FLIP ? K * K - 1 - tap : tap);
    weights[i] = make_float2(__bfloat162float(weight[source]), __bfloat162float(weight[source + K * K]));
  }

  const int pair = threadIdx.x % kPairs;
  const int row0 = ((threadIdx.x / kPairs) % (kPipeTileRows / kPipeRows)) * kPipeRows;
  const int strip = threadIdx.x / (kPairs * (kPipeTileRows / kPipeRows));
  const float2 b = bias != nullptr ? load_pair(bias + cp0 + pair) : make_float2(0.0f, 0.0f);

  auto tile_origin = [&](int64_t t, int64_t& n, int& oh0, int& ow0) {
    ow0 = (t % tiles_w) * TCOLS;
    oh0 = ((t / tiles_w) % tiles_h) * kPipeTileRows;
    n = t / (static_cast<int64_t>(tiles_w) * tiles_h);
  };

  int64_t t = blockIdx.x;
  if (t < tiles) {
    int64_t n;
    int oh0, ow0;
    tile_origin(t, n, oh0, ow0);
    issue_tile<Shape::rows, Shape::cols, Shape::stride>(buffers, x, n, oh0 - P, ow0 - P, cp0, height, width, pairs);
  }
  cp_async_commit();
  for (int buffer = 0; t < tiles; t += gridDim.x, buffer ^= 1) {
    const int64_t next = t + gridDim.x;
    if (next < tiles) {
      int64_t n;
      int oh0, ow0;
      tile_origin(next, n, oh0, ow0);
      issue_tile<Shape::rows, Shape::cols, Shape::stride>(buffers + (buffer ^ 1) * Shape::buffer_words, x, n, oh0 - P,
                                                          ow0 - P, cp0, height, width, pairs);
    }
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();

    int64_t n;
    int oh0, ow0;
    tile_origin(t, n, oh0, ow0);
    const __nv_bfloat162* tile = buffers + buffer * Shape::buffer_words;
    float2 acc[kPipeRows][kPipeStrip];
#pragma unroll
    for (int r = 0; r < kPipeRows; ++r) {
#pragma unroll
      for (int j = 0; j < kPipeStrip; ++j) {
        acc[r][j] = b;
      }
    }
#pragma unroll
    for (int ir = 0; ir < kPipeRows + K - 1; ++ir) {
      const __nv_bfloat162* source = tile + (row0 + ir) * Shape::stride + strip * kPipeStrip * kPairs + pair;
      float2 v[kPipeStrip + K - 1];
#pragma unroll
      for (int j = 0; j < kPipeStrip + K - 1; ++j) {
        v[j] = __bfloat1622float2(source[j * kPairs]);
      }
#pragma unroll
      for (int r = 0; r < kPipeRows; ++r) {
        const int kh = ir - r;
        if (kh < 0 || kh >= K) {
          continue;
        }
#pragma unroll
        for (int kw = 0; kw < K; ++kw) {
          const float2 wk = weights[(kh * K + kw) * kPairs + pair];
#pragma unroll
          for (int j = 0; j < kPipeStrip; ++j) {
            acc[r][j].x = fmaf(v[j + kw].x, wk.x, acc[r][j].x);
            acc[r][j].y = fmaf(v[j + kw].y, wk.y, acc[r][j].y);
          }
        }
      }
    }
    const int ow = ow0 + strip * kPipeStrip;
#pragma unroll
    for (int r = 0; r < kPipeRows; ++r) {
      const int oh = oh0 + row0 + r;
      if (oh >= height) {
        continue;
      }
      __nv_bfloat162* out = y + ((n * height + oh) * width + ow) * pairs + cp0 + pair;
#pragma unroll
      for (int j = 0; j < kPipeStrip; ++j) {
        if (ow + j < width) {
          out[static_cast<int64_t>(j) * pairs] = __float22bfloat162_rn(acc[r][j]);
        }
      }
    }
    __syncthreads();
  }
  cp_async_wait<0>();
}

// Weight gradient with the same pipeline: a thread keeps the K taps of one kernel row of one channel pair and slides a
// register window along each of its tile rows. Threads a kernel row apart sit a half-warp apart, so the tiles use
// padded_stride (% 32 == 16). Partials are [blockIdx.x * kGradRowGroups + group][K * K][pairs] float2 and, from the
// kernel-row-0 threads, [blockIdx.x * kGradRowGroups + group][pairs] float2 for the bias.
template <int K, int TCOLS>
struct GradShape {
  static constexpr int rows = kPipeTileRows + K - 1;
  static constexpr int cols = TCOLS + K - 1;
  static constexpr int stride = padded_stride(cols * kPairs);
  static constexpr int grad_stride = padded_stride(TCOLS * kPairs);
  static constexpr int x_words = rows * stride;
  static constexpr int buffer_words = x_words + kPipeTileRows * grad_stride;
  static constexpr int bytes = 2 * buffer_words * 4;
  static constexpr int threads = kPairs * K * kGradRowGroups;
};

template <int K, int TCOLS>
__global__ void __launch_bounds__(GradShape<K, TCOLS>::threads, 2)
    depthwise_conv_weight_grad_pipelined_kernel(const __nv_bfloat162* __restrict__ x,
                                                const __nv_bfloat162* __restrict__ grad,
                                                float2* __restrict__ weight_partial, float2* __restrict__ bias_partial,
                                                int batch, int height, int width, int pairs) {
  using Shape = GradShape<K, TCOLS>;
  constexpr int P = K / 2;
  extern __shared__ __align__(16) unsigned char shared[];
  auto* buffers = reinterpret_cast<__nv_bfloat162*>(shared);

  const int pair = threadIdx.x % kPairs;
  const int kh = (threadIdx.x / kPairs) % K;
  const int group = threadIdx.x / (kPairs * K);
  const int cp0 = blockIdx.y * kPairs;
  const int tiles_w = (width + TCOLS - 1) / TCOLS;
  const int tiles_h = (height + kPipeTileRows - 1) / kPipeTileRows;
  const int64_t tiles = static_cast<int64_t>(batch) * tiles_h * tiles_w;

  auto issue = [&](int64_t t, __nv_bfloat162* buffer) {
    const int ow0 = (t % tiles_w) * TCOLS;
    const int oh0 = ((t / tiles_w) % tiles_h) * kPipeTileRows;
    const int64_t n = t / (static_cast<int64_t>(tiles_w) * tiles_h);
    issue_tile<Shape::rows, Shape::cols, Shape::stride>(buffer, x, n, oh0 - P, ow0 - P, cp0, height, width, pairs);
    issue_tile<kPipeTileRows, TCOLS, Shape::grad_stride>(buffer + Shape::x_words, grad, n, oh0, ow0, cp0, height, width,
                                                         pairs);
  };

  float2 acc[K];
#pragma unroll
  for (int kw = 0; kw < K; ++kw) {
    acc[kw] = make_float2(0.0f, 0.0f);
  }
  float2 bias_acc = make_float2(0.0f, 0.0f);

  int64_t t = blockIdx.x;
  if (t < tiles) {
    issue(t, buffers);
  }
  cp_async_commit();
  for (int buffer = 0; t < tiles; t += gridDim.x, buffer ^= 1) {
    if (t + gridDim.x < tiles) {
      issue(t + gridDim.x, buffers + (buffer ^ 1) * Shape::buffer_words);
    }
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    const __nv_bfloat162* tile = buffers + buffer * Shape::buffer_words;
    for (int r = group; r < kPipeTileRows; r += kGradRowGroups) {
      const __nv_bfloat162* source = tile + (r + kh) * Shape::stride + pair;
      const __nv_bfloat162* g_row = tile + Shape::x_words + r * Shape::grad_stride + pair;
      float2 window[K];
#pragma unroll
      for (int q = 0; q < K - 1; ++q) {
        window[q] = __bfloat1622float2(source[q * kPairs]);
      }
#pragma unroll
      for (int j = 0; j < TCOLS; ++j) {
        window[K - 1] = __bfloat1622float2(source[(j + K - 1) * kPairs]);
        const float2 g = __bfloat1622float2(g_row[j * kPairs]);
#pragma unroll
        for (int kw = 0; kw < K; ++kw) {
          acc[kw].x = fmaf(window[kw].x, g.x, acc[kw].x);
          acc[kw].y = fmaf(window[kw].y, g.y, acc[kw].y);
        }
        if (kh == 0) {
          bias_acc.x += g.x;
          bias_acc.y += g.y;
        }
#pragma unroll
        for (int q = 0; q < K - 1; ++q) {
          window[q] = window[q + 1];
        }
      }
    }
    __syncthreads();
  }
  cp_async_wait<0>();
  const int64_t slot = static_cast<int64_t>(blockIdx.x) * kGradRowGroups + group;
  const int cp = cp0 + pair;
#pragma unroll
  for (int kw = 0; kw < K; ++kw) {
    weight_partial[(slot * K * K + kh * K + kw) * pairs + cp] = acc[kw];
  }
  if (kh == 0) {
    bias_partial[slot * pairs + cp] = bias_acc;
  }
}

// Sums the weight-gradient partials [slots][taps][C] and bias partials [slots][C] over the slots, in slot order, and
// writes [C][taps] and [C] in bf16: one thread per output, reading along the channels.
__global__ void depthwise_conv_grad_reduce_kernel(const float* __restrict__ weight_partial,
                                                  const float* __restrict__ bias_partial,
                                                  __nv_bfloat16* __restrict__ grad_weight,
                                                  __nv_bfloat16* __restrict__ grad_bias, int slots, int channels,
                                                  int taps) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= channels * (taps + 1)) {
    return;
  }
  const int c = index % channels;
  const int tap = index / channels;
  float sum = 0.0f;
  if (tap < taps) {
    for (int slot = 0; slot < slots; ++slot) {
      sum += weight_partial[(static_cast<int64_t>(slot) * taps + tap) * channels + c];
    }
    grad_weight[static_cast<int64_t>(c) * taps + tap] = __float2bfloat16_rn(sum);
  } else {
    for (int slot = 0; slot < slots; ++slot) {
      sum += bias_partial[static_cast<int64_t>(slot) * channels + c];
    }
    grad_bias[c] = __float2bfloat16_rn(sum);
  }
}

void check_inputs(const at::Tensor& x, const at::Tensor& weight) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 4, "depthwise_conv: x must be a 4-D CUDA tensor");
  TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast), "depthwise_conv: x must be channels-last");
  TORCH_CHECK(x.numel() < (int64_t{1} << 31), "depthwise_conv: x is too large");
  const int64_t channels = x.size(1);
  TORCH_CHECK(
      weight.dim() == 4 && weight.size(0) == channels && weight.size(1) == 1 && weight.size(2) == weight.size(3),
      "depthwise_conv: weight must be [C, 1, k, k]");
  TORCH_CHECK(weight.size(2) % 2 == 1, "depthwise_conv: the kernel size must be odd");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "depthwise_conv: weight and x must share a dtype");
}

bool use_tiled_kernels(const at::Tensor& x, const at::Tensor& weight) {
  const int64_t kernel = weight.size(2);
  return x.scalar_type() == at::kBFloat16 && x.size(1) % 2 == 0 && (kernel == 3 || kernel == 5 || kernel == 7);
}

const __nv_bfloat162* pairs_of(const at::Tensor& t) {
  return reinterpret_cast<const __nv_bfloat162*>(t.data_ptr<at::BFloat16>());
}

// Launches the tiled convolution kernel with tile width TW.
template <int K, int TW, bool FLIP>
void launch_conv_tiled(const at::Tensor& x, const at::Tensor& weight_t, const at::Tensor* bias, at::Tensor& y) {
  const int height = static_cast<int>(x.size(2));
  const int width = static_cast<int>(x.size(3));
  const int pairs = static_cast<int>(x.size(1) / 2);
  const int64_t tiles = x.size(0) * ((height + kTileRows - 1) / kTileRows) * ((width + TW - 1) / TW);
  const dim3 grid(static_cast<unsigned>(tiles), (pairs + kPairs - 1) / kPairs);
  constexpr int threads = kPairs * kTileRows * (TW / kStrip);
  depthwise_conv_tiled_kernel<K, TW, FLIP><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      pairs_of(x), pairs_of(weight_t), bias != nullptr ? pairs_of(*bias) : nullptr,
      reinterpret_cast<__nv_bfloat162*>(y.data_ptr<at::BFloat16>()), height, width, pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int K, bool FLIP>
void launch_conv_tiled(const at::Tensor& x, const at::Tensor& weight_t, const at::Tensor* bias, at::Tensor& y) {
  if (x.size(3) > 16) {
    launch_conv_tiled<K, 32, FLIP>(x, weight_t, bias, y);
  } else {
    launch_conv_tiled<K, 16, FLIP>(x, weight_t, bias, y);
  }
}

// The tiled convolution of channels-last bf16 x with a [C, 1, k, k] weight, k 3, 5 or 7.
template <bool FLIP>
void launch_conv_tiled(const at::Tensor& x, const at::Tensor& weight, const at::Tensor* bias, at::Tensor& y) {
  // [C, 1, k, k] -> [k * k, C], so one bf16x2 load reads a tap of two neighbouring channels.
  const auto weight_t = weight.reshape({weight.size(0), -1}).t().contiguous();
  switch (weight.size(2)) {
    case 3:
      launch_conv_tiled<3, FLIP>(x, weight_t, bias, y);
      break;
    case 5:
      launch_conv_tiled<5, FLIP>(x, weight_t, bias, y);
      break;
    default:
      launch_conv_tiled<7, FLIP>(x, weight_t, bias, y);
  }
}

bool use_pipelined_kernels(const at::Tensor& x, const at::Tensor& weight) {
  return use_tiled_kernels(x, weight) && x.size(1) % (2 * kPairs) == 0 &&
         reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0;
}

// Raises a kernel's dynamic shared memory limit once, before its first launch (outside any graph capture).
template <auto Kernel>
void allow_shared_memory(int bytes) {
  static const bool configured = [bytes] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    return true;
  }();
  (void)configured;
}

int resident_blocks(int per_sm) { return at::cuda::getCurrentDeviceProperties()->multiProcessorCount * per_sm; }

// Launches the pipelined convolution kernel: enough blocks per 32-channel chunk to fill the SMs, at most one per tile.
template <int K, int TCOLS, bool FLIP>
void launch_conv_pipelined(const at::Tensor& x, const at::Tensor& weight, const at::Tensor* bias, at::Tensor& y) {
  using Shape = PipeShape<K, TCOLS>;
  auto* kernel = depthwise_conv_pipelined_kernel<K, TCOLS, FLIP>;
  allow_shared_memory<depthwise_conv_pipelined_kernel<K, TCOLS, FLIP>>(Shape::bytes);
  const int batch = static_cast<int>(x.size(0));
  const int height = static_cast<int>(x.size(2));
  const int width = static_cast<int>(x.size(3));
  const int pairs = static_cast<int>(x.size(1) / 2);
  const int chunks = pairs / kPairs;
  const int64_t tiles =
      static_cast<int64_t>(batch) * ((height + kPipeTileRows - 1) / kPipeTileRows) * ((width + TCOLS - 1) / TCOLS);
  const int resident = resident_blocks(kPipeBlocksPerSm * (32 / TCOLS));
  const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(tiles, (resident + chunks - 1) / chunks)));
  kernel<<<dim3(blocks, chunks), pipe_threads(TCOLS), Shape::bytes, at::cuda::getCurrentCUDAStream()>>>(
      pairs_of(x), reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
      bias != nullptr ? pairs_of(*bias) : nullptr, reinterpret_cast<__nv_bfloat162*>(y.data_ptr<at::BFloat16>()), batch,
      height, width, pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int K, bool FLIP>
void launch_conv_pipelined(const at::Tensor& x, const at::Tensor& weight, const at::Tensor* bias, at::Tensor& y) {
  if (x.size(3) > 16) {
    launch_conv_pipelined<K, 32, FLIP>(x, weight, bias, y);
  } else {
    launch_conv_pipelined<K, 16, FLIP>(x, weight, bias, y);
  }
}

// The pipelined convolution of channels-last bf16 x with a [C, 1, k, k] weight, k 3, 5 or 7.
template <bool FLIP>
void launch_conv_pipelined(const at::Tensor& x, const at::Tensor& weight, const at::Tensor* bias, at::Tensor& y) {
  const auto weight_c = weight.contiguous();
  switch (weight.size(2)) {
    case 3:
      launch_conv_pipelined<3, FLIP>(x, weight_c, bias, y);
      break;
    case 5:
      launch_conv_pipelined<5, FLIP>(x, weight_c, bias, y);
      break;
    default:
      launch_conv_pipelined<7, FLIP>(x, weight_c, bias, y);
  }
}

// The convolution, or with flip its transpose (the kernel rotated by 180 degrees), which is the input gradient.
at::Tensor launch_conv(const at::Tensor& x, const at::Tensor& weight, const at::Tensor* bias, bool flip) {
  auto y = at::empty_like(x, x.options(), at::MemoryFormat::ChannelsLast);
  const int64_t total = x.numel();
  if (total == 0) {
    return y;
  }
  if (depthwise_tc::supported(x, weight)) {
    return depthwise_tc::conv(x, weight, bias, flip);
  }
  if (use_pipelined_kernels(x, weight)) {
    if (flip) {
      launch_conv_pipelined<true>(x, weight, bias, y);
    } else {
      launch_conv_pipelined<false>(x, weight, bias, y);
    }
    return y;
  }
  if (use_tiled_kernels(x, weight)) {
    if (flip) {
      launch_conv_tiled<true>(x, weight, bias, y);
    } else {
      launch_conv_tiled<false>(x, weight, bias, y);
    }
    return y;
  }
  const auto weight_c = flip ? weight.flip({2, 3}).contiguous() : weight.contiguous();
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "depthwise_conv", [&] {
    depthwise_conv_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
        x.data_ptr<scalar_t>(), weight_c.data_ptr<scalar_t>(), bias != nullptr ? bias->data_ptr<scalar_t>() : nullptr,
        y.data_ptr<scalar_t>(), total, static_cast<int>(x.size(2)), static_cast<int>(x.size(3)),
        static_cast<int>(x.size(1)), static_cast<int>(weight.size(2)));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// Forward op: channels-last x [N, C, H, W], weight [C, 1, k, k] and optional bias [C], all in x's dtype.
at::Tensor depthwise_conv(const at::Tensor& x, const at::Tensor& weight, const std::optional<at::Tensor>& bias) {
  check_inputs(x, weight);
  const c10::cuda::CUDAGuard guard(x.device());
  if (bias.has_value()) {
    TORCH_CHECK(bias->numel() == x.size(1) && bias->scalar_type() == x.scalar_type(),
                "depthwise_conv: bias must be [C] in x's dtype");
    const auto bias_c = bias->contiguous();
    return launch_conv(x, weight, &bias_c, false);
  }
  return launch_conv(x, weight, nullptr, false);
}

// Launches the tiled weight-gradient kernel into blocks * kGradRowGroups slots of partials.
template <int K, int TW>
void launch_weight_grad_tiled(const at::Tensor& x, const at::Tensor& grad, at::Tensor& weight_partial,
                              at::Tensor& bias_partial, int blocks) {
  const int pairs = static_cast<int>(x.size(1) / 2);
  const dim3 grid(blocks, (pairs + kPairs - 1) / kPairs);
  constexpr int threads = kPairs * K * kGradRowGroups;
  depthwise_conv_weight_grad_tiled_kernel<K, TW><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      pairs_of(x), pairs_of(grad), reinterpret_cast<float2*>(weight_partial.data_ptr<float>()),
      reinterpret_cast<float2*>(bias_partial.data_ptr<float>()), static_cast<int>(x.size(0)),
      static_cast<int>(x.size(2)), static_cast<int>(x.size(3)), pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int K>
void launch_weight_grad_tiled(const at::Tensor& x, const at::Tensor& grad, at::Tensor& weight_partial,
                              at::Tensor& bias_partial, int blocks) {
  if (x.size(3) > 16) {
    launch_weight_grad_tiled<K, 32>(x, grad, weight_partial, bias_partial, blocks);
  } else {
    launch_weight_grad_tiled<K, 16>(x, grad, weight_partial, bias_partial, blocks);
  }
}

// The weight and bias gradients through the tiled kernel: [C, 1, k, k] and [C] in fp32.
std::tuple<at::Tensor, at::Tensor> weight_grad_tiled(const at::Tensor& x, const at::Tensor& grad, int kernel) {
  const int64_t channels = x.size(1);
  const int64_t chunks = (channels / 2 + kPairs - 1) / kPairs;
  const int64_t tile_width = x.size(3) > 16 ? 32 : 16;
  const int64_t tiles =
      x.size(0) * ((x.size(2) + kGradTileRows - 1) / kGradTileRows) * ((x.size(3) + tile_width - 1) / tile_width);
  const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(tiles, kGradTargetBlocks / chunks)));
  const int64_t slots = static_cast<int64_t>(blocks) * kGradRowGroups;
  auto float_options = x.options().dtype(at::kFloat);
  auto weight_partial = at::zeros({slots, kernel * kernel, channels}, float_options);
  auto bias_partial = at::zeros({slots, channels}, float_options);
  switch (kernel) {
    case 3:
      launch_weight_grad_tiled<3>(x, grad, weight_partial, bias_partial, blocks);
      break;
    case 5:
      launch_weight_grad_tiled<5>(x, grad, weight_partial, bias_partial, blocks);
      break;
    default:
      launch_weight_grad_tiled<7>(x, grad, weight_partial, bias_partial, blocks);
  }
  auto grad_weight = weight_partial.sum(0).t().reshape({channels, 1, kernel, kernel});
  return {grad_weight, bias_partial.sum(0)};
}

// The pipelined weight-gradient kernel and its slot reduction into bf16 grad_weight [C, 1, k, k] and grad_bias [C].
template <int K, int TCOLS>
void weight_grad_pipelined(const at::Tensor& x, const at::Tensor& grad, at::Tensor& grad_weight,
                           at::Tensor& grad_bias) {
  using Shape = GradShape<K, TCOLS>;
  auto* kernel = depthwise_conv_weight_grad_pipelined_kernel<K, TCOLS>;
  allow_shared_memory<depthwise_conv_weight_grad_pipelined_kernel<K, TCOLS>>(Shape::bytes);
  const int batch = static_cast<int>(x.size(0));
  const int height = static_cast<int>(x.size(2));
  const int width = static_cast<int>(x.size(3));
  const int channels = static_cast<int>(x.size(1));
  const int pairs = channels / 2;
  const int chunks = pairs / kPairs;
  const int64_t tiles =
      static_cast<int64_t>(batch) * ((height + kPipeTileRows - 1) / kPipeTileRows) * ((width + TCOLS - 1) / TCOLS);
  const int resident = resident_blocks(2 * (32 / TCOLS));
  const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(tiles, (resident + chunks - 1) / chunks)));
  const int slots = blocks * kGradRowGroups;
  auto float_options = x.options().dtype(at::kFloat);
  auto weight_partial = at::empty({slots, K * K, channels}, float_options);
  auto bias_partial = at::empty({slots, channels}, float_options);
  auto stream = at::cuda::getCurrentCUDAStream();
  kernel<<<dim3(blocks, chunks), Shape::threads, Shape::bytes, stream>>>(
      pairs_of(x), pairs_of(grad), reinterpret_cast<float2*>(weight_partial.data_ptr<float>()),
      reinterpret_cast<float2*>(bias_partial.data_ptr<float>()), batch, height, width, pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int outputs = channels * (K * K + 1);
  depthwise_conv_grad_reduce_kernel<<<(outputs + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      weight_partial.data_ptr<float>(), bias_partial.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(grad_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(grad_bias.data_ptr<at::BFloat16>()), slots, channels, K * K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int K>
void weight_grad_pipelined(const at::Tensor& x, const at::Tensor& grad, at::Tensor& grad_weight,
                           at::Tensor& grad_bias) {
  if (x.size(3) > 16) {
    weight_grad_pipelined<K, 32>(x, grad, grad_weight, grad_bias);
  } else {
    weight_grad_pipelined<K, 16>(x, grad, grad_weight, grad_bias);
  }
}

// Launches the fallback weight-gradient kernel over chunks x channel_blocks blocks.
template <int KERNEL>
void launch_weight_grad(const at::Tensor& x, const at::Tensor& grad, at::Tensor& weight_partial,
                        at::Tensor& bias_partial, int64_t pixels_per_chunk, int chunks, int channel_blocks) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(chunks, channel_blocks);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "depthwise_conv_weight_grad", [&] {
    depthwise_conv_weight_grad_kernel<scalar_t, KERNEL><<<grid, kThreads, 0, stream>>>(
        x.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), weight_partial.data_ptr<float>(),
        bias_partial.data_ptr<float>(), x.size(0) * x.size(2) * x.size(3), pixels_per_chunk,
        static_cast<int>(x.size(2)), static_cast<int>(x.size(3)), static_cast<int>(x.size(1)));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Backward op: the gradients of x (channels-last), weight [C, 1, k, k] and bias [C], in x's dtype.
std::tuple<at::Tensor, at::Tensor, at::Tensor> depthwise_conv_backward(const at::Tensor& grad_in, const at::Tensor& x,
                                                                       const at::Tensor& weight) {
  check_inputs(x, weight);
  const c10::cuda::CUDAGuard guard(x.device());
  const auto grad = grad_in.contiguous(at::MemoryFormat::ChannelsLast);
  TORCH_CHECK(grad.sizes() == x.sizes() && grad.scalar_type() == x.scalar_type(), "depthwise_conv: bad grad");

  // With stride 1 and padding k / 2, the input gradient is the same convolution with the kernel flipped.
  auto grad_x = launch_conv(grad, weight, nullptr, true);

  const int kernel = static_cast<int>(weight.size(2));
  if (depthwise_tc::supported(x, weight) && reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0) {
    auto [grad_weight, grad_bias] = depthwise_tc::weight_grad(x, grad, weight);
    return {grad_x, grad_weight, grad_bias};
  }
  if (use_pipelined_kernels(x, weight) && reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0) {
    auto grad_weight = at::empty_like(weight, at::MemoryFormat::Contiguous);
    auto grad_bias = at::empty({x.size(1)}, weight.options());
    if (kernel == 3) {
      weight_grad_pipelined<3>(x, grad, grad_weight, grad_bias);
    } else if (kernel == 5) {
      weight_grad_pipelined<5>(x, grad, grad_weight, grad_bias);
    } else {
      weight_grad_pipelined<7>(x, grad, grad_weight, grad_bias);
    }
    return {grad_x, grad_weight, grad_bias};
  }
  if (use_tiled_kernels(x, weight) && x.numel() > 0) {
    auto [grad_weight, grad_bias] = weight_grad_tiled(x, grad, kernel);
    return {grad_x, grad_weight.to(weight.scalar_type()), grad_bias.to(weight.scalar_type())};
  }
  const int64_t channels = x.size(1);
  const int64_t pixels = x.size(0) * x.size(2) * x.size(3);
  const int channel_blocks = static_cast<int>((channels + kThreads - 1) / kThreads);
  int64_t chunks = std::max<int64_t>(1, std::min<int64_t>(pixels, kWeightGradBlocks / channel_blocks));
  const int64_t pixels_per_chunk = (pixels + chunks - 1) / chunks;
  chunks = std::max<int64_t>(1, (pixels + pixels_per_chunk - 1) / pixels_per_chunk);
  auto float_options = x.options().dtype(at::kFloat);
  auto weight_partial = at::empty({chunks, channels, kernel * kernel}, float_options);
  auto bias_partial = at::empty({chunks, channels}, float_options);
  if (pixels > 0) {
    switch (kernel) {
      case 3:
        launch_weight_grad<3>(x, grad, weight_partial, bias_partial, pixels_per_chunk, chunks, channel_blocks);
        break;
      case 5:
        launch_weight_grad<5>(x, grad, weight_partial, bias_partial, pixels_per_chunk, chunks, channel_blocks);
        break;
      case 7:
        launch_weight_grad<7>(x, grad, weight_partial, bias_partial, pixels_per_chunk, chunks, channel_blocks);
        break;
      default:
        TORCH_CHECK(false, "depthwise_conv: the weight gradient supports kernel sizes 3, 5 and 7, got ", kernel);
    }
  } else {
    weight_partial.zero_();
    bias_partial.zero_();
  }
  auto grad_weight = weight_partial.sum(0).view(weight.sizes()).to(weight.scalar_type());
  auto grad_bias = bias_partial.sum(0).to(weight.scalar_type());
  return {grad_x, grad_weight, grad_bias};
}

}  // namespace

TORCH_LIBRARY_IMPL(py123d_garage_cuda, CUDA, m) {
  m.impl("depthwise_conv", &depthwise_conv);
  m.impl("depthwise_conv_backward", &depthwise_conv_backward);
}

}  // namespace py123d_garage_cuda
