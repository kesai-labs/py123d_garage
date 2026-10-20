// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// ConvNeXt's depthwise 7 x 7 convolution (stride 1, padding 3) on Hopper tensor cores for bf16 channels-last tensors
// with C % 16 == 0: the forward, the input gradient (kernel rotated by 180 degrees) and the weight and bias gradients.
//
// Each channel is a Toeplitz product: per kernel row kh, 16 output rows x 8 columns are one mma.m16n8k16 over a
// 16-column input window starting 4 columns left of the outputs,
//   y[h][j] += sum_k x[h + kh - 3][j0 - 4 + k] * w[kh][k - j - 1]      (B[k][j] zero outside 0 <= k - j - 1 < 7)
// and the weight gradient swaps the operands' roles,
//   D_kh[i][j] += sum_h x[h + kh - 3][j0 - 4 + i] * g[h][j0 + j],   w_grad[kh][i - j - 1] = sum of D_kh over i - j.
// A block (8 warps, 16 channels) loads channels-last boxes by TMA and transposes them into per-channel planes.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include "transfuser/csrc/sm80/ptx.cuh"
#include "transfuser/csrc/sm90/depthwise_conv_tc.h"
#include "transfuser/csrc/sm90/tma.cuh"

namespace py123d_garage_cuda {
namespace depthwise_tc {
namespace {

constexpr int K = 7;
constexpr int P = K / 2;
constexpr int kRows = 16;               // output rows per tile: the MMA's M (forward) or K (weight gradient)
constexpr int kInRows = kRows + K - 1;  // input rows per tile
constexpr int kChannels = 16;           // channels per block
constexpr int kChunks = kChannels / 8;  // 16-byte chunks per pixel in a box
constexpr int kWarps = 8;
constexpr int kPerWarp = kChannels / kWarps;
constexpr int kThreads = 32 * kWarps;
constexpr int kBlocksPerSm = 2;

// Shared-memory layout of a tile TW output columns wide.
template <int TW>
struct Tile {
  static constexpr int cols = TW + 8;  // input columns [ow0 - 4, ow0 + TW + 4)
  static constexpr int pixels = kInRows * cols;
  static constexpr int box_bytes = pixels * kChannels * 2;
  // Plane row stride: an odd number of 16-byte units, so ldmatrix's 8 rows hit 8 distinct bank groups.
  static constexpr int row_bytes = ((cols * 2 / 16) | 1) * 16;
  static constexpr int plane_bytes = kInRows * row_bytes;
  static constexpr int planes_bytes = kChannels * plane_bytes;
  // Output (or gradient) tiles: kRows x TW, no halo.
  static constexpr int out_pixels = kRows * TW;
  static constexpr int out_box_bytes = out_pixels * kChannels * 2;
  static constexpr int out_row_bytes = ((TW * 2 / 16) | 1) * 16;
  static constexpr int out_plane_bytes = kRows * out_row_bytes;
  static constexpr int out_planes_bytes = kChannels * out_plane_bytes;
  // Forward: input box, planes (reused for the outputs), output box, barrier.
  static constexpr int conv_bytes = box_bytes + planes_bytes + out_box_bytes + 16;
  // Weight gradient: input box and planes, gradient box and planes, barrier.
  static constexpr int grad_bytes = box_bytes + planes_bytes + out_box_bytes + out_planes_bytes + 16;
};

// Channels-last box (pixels x 16 channels, `cols` pixels per row) -> 16 planes, two horizontally adjacent pixels per
// thread so every store is one 32-bit word of two columns.
__device__ __forceinline__ void box_to_planes(const char* box, char* planes, int pixels, int cols, int row_bytes,
                                              int plane_bytes) {
  for (int i = threadIdx.x; i < pixels / 2 * kChunks; i += kThreads) {
    const int pixel = 2 * (i / kChunks);
    const int chunk = i % kChunks;
    const uint4 v0 = *reinterpret_cast<const uint4*>(box + (pixel * kChunks + chunk) * 16);
    const uint4 v1 = *reinterpret_cast<const uint4*>(box + ((pixel + 1) * kChunks + chunk) * 16);
    const unsigned w0[4] = {v0.x, v0.y, v0.z, v0.w};
    const unsigned w1[4] = {v1.x, v1.y, v1.z, v1.w};
    char* dst = planes + chunk * 8 * plane_bytes + (pixel / cols) * row_bytes + (pixel % cols) * 2;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      *reinterpret_cast<unsigned*>(dst + c * plane_bytes) = __byte_perm(w0[c / 2], w1[c / 2], c % 2 ? 0x7632 : 0x5410);
    }
  }
}

// The inverse of box_to_planes for a tile of kRows x TW pixels.
template <int TW>
__device__ __forceinline__ void planes_to_box(const char* planes, char* box, int row_bytes, int plane_bytes) {
  for (int i = threadIdx.x; i < kRows * TW / 2 * kChunks; i += kThreads) {
    const int pixel = 2 * (i / kChunks);
    const int chunk = i % kChunks;
    const char* src = planes + chunk * 8 * plane_bytes + (pixel / TW) * row_bytes + (pixel % TW) * 2;
    unsigned p0[4], p1[4];
#pragma unroll
    for (int c = 0; c < 8; c += 2) {
      const unsigned even = *reinterpret_cast<const unsigned*>(src + c * plane_bytes);
      const unsigned odd = *reinterpret_cast<const unsigned*>(src + (c + 1) * plane_bytes);
      p0[c / 2] = __byte_perm(even, odd, 0x5410);
      p1[c / 2] = __byte_perm(even, odd, 0x7632);
    }
    char* dst = box + (pixel * kChunks + chunk) * 16;
    *reinterpret_cast<uint4*>(dst) = make_uint4(p0[0], p0[1], p0[2], p0[3]);
    *reinterpret_cast<uint4*>(dst + kChunks * 16) = make_uint4(p1[0], p1[1], p1[2], p1[3]);
  }
}

template <int TW>
__device__ __forceinline__ void tile_origin(int64_t t, int tiles_w, int tiles_h, int& n, int& oh0, int& ow0) {
  ow0 = static_cast<int>(t % tiles_w) * TW;
  oh0 = static_cast<int>((t / tiles_w) % tiles_h) * kRows;
  n = static_cast<int>(t / (static_cast<int64_t>(tiles_w) * tiles_h));
}

// y = conv2d(x, weight, bias) with a [C, 1, 7, 7] weight, with FLIP rotated by 180 degrees. Resident blocks loop over
// the tiles of one 16-channel chunk (blockIdx.y), overlapping the next tile's TMA load and the previous tile's store
// with the MMAs.
template <int TW, bool FLIP>
__global__ void __launch_bounds__(kThreads, kBlocksPerSm)
    conv_kernel(const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap y_map,
                const __nv_bfloat16* __restrict__ weight,  // [C, 1, K, K]
                const __nv_bfloat16* __restrict__ bias, int batch, int height, int width) {
  using T = Tile<TW>;
  constexpr int kTilesN = TW / 8;
  extern __shared__ __align__(128) char shared[];
  char* box = shared;
  char* planes = box + T::box_bytes;
  char* out_box = planes + T::planes_bytes;
  const unsigned bar = shared_address(out_box + T::out_box_bytes);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int c0 = blockIdx.y * kChannels;

  // B[k][j] = w[kh][k - j - 1]: the lane holds k = 2 (lane % 4) + {0, 1} (b0) and + 8 (b1) of column j = lane / 4.
  unsigned b[kPerWarp][K][2];
  float bias_c[kPerWarp];
#pragma unroll
  for (int p = 0; p < kPerWarp; ++p) {
    const int channel = c0 + warp + kWarps * p;
    const __nv_bfloat16* w_c = weight + static_cast<int64_t>(channel) * K * K;
#pragma unroll
    for (int kh = 0; kh < K; ++kh) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        unsigned packed = 0;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int kw = 2 * (lane % 4) + e + 8 * half - lane / 4 - 1;
          if (kw >= 0 && kw < K) {
            const int tap = FLIP ? (K - 1 - kh) * K + (K - 1 - kw) : kh * K + kw;
            packed |= static_cast<unsigned>(__bfloat16_as_ushort(w_c[tap])) << (16 * e);
          }
        }
        b[p][kh][half] = packed;
      }
    }
    bias_c[p] = bias != nullptr ? __bfloat162float(bias[channel]) : 0.0f;
  }

  const int tiles_w = (width + TW - 1) / TW;
  const int tiles_h = (height + kRows - 1) / kRows;
  const int64_t tiles = static_cast<int64_t>(batch) * tiles_h * tiles_w;
  auto issue = [&](int64_t t) {
    int n, oh0, ow0;
    tile_origin<TW>(t, tiles_w, tiles_h, n, oh0, ow0);
    mbarrier_expect(bar, T::box_bytes);
    tma_load(shared_address(box), &x_map, c0, ow0 - 4, oh0 - P, n, bar);
  };

  // ldmatrix rows: lanes 0-15 address rows 0-15 of a window's first 8 columns, lanes 16-31 those of the next 8.
  const unsigned lane_offset = (lane & 15) * T::row_bytes + (lane >> 4) * 16;

  int64_t t = blockIdx.x;
  if (threadIdx.x == 0) {
    mbarrier_init(bar, 1);
    fence_mbarrier_init();
    if (t < tiles) {
      issue(t);
    }
  }
  __syncthreads();
  for (unsigned parity = 0; t < tiles; t += gridDim.x, parity ^= 1) {
    mbarrier_wait(bar, parity);
    box_to_planes(box, planes, T::pixels, T::cols, T::row_bytes, T::plane_bytes);
    __syncthreads();
    if (threadIdx.x == 0 && t + gridDim.x < tiles) {
      issue(t + gridDim.x);
    }

#pragma unroll
    for (int p = 0; p < kPerWarp; ++p) {
      const int plane = warp + kWarps * p;
      const unsigned rows = shared_address(planes + plane * T::plane_bytes) + lane_offset;
      float acc[kTilesN][4];
#pragma unroll
      for (int j = 0; j < kTilesN; ++j) {
        acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = bias_c[p];
      }
#pragma unroll
      for (int kh = 0; kh < K; ++kh) {
        const unsigned row = rows + kh * T::row_bytes;
        // Matrices by lane groups of 8: (rows 0-7, cols 0-7), (rows 8-15, cols 0-7), (rows 0-7, cols 8-15),
        // (rows 8-15, cols 8-15) = a0..a3 of the A fragment.
        unsigned a[4];
        ldmatrix_x4(a, row);
        mma(acc[0], a, b[p][kh][0], b[p][kh][1]);
#pragma unroll
        for (int j = 1; j < kTilesN; ++j) {
          // Tile j's window, columns 8j..8j+15, is the previous window's upper half plus 8 new columns.
          a[0] = a[2];
          a[1] = a[3];
          ldmatrix_x2(a[2], a[3], row + (j + 1) * 16 - 16 * (lane >> 4));
          mma(acc[j], a, b[p][kh][0], b[p][kh][1]);
        }
      }
      __syncwarp();  // only this warp reads its plane: the outputs overwrite its rows 0-15
      char* out_plane = planes + plane * T::plane_bytes;
      const int r = lane / 4;
#pragma unroll
      for (int j = 0; j < kTilesN; ++j) {
        const int col = 8 * j + 2 * (lane % 4);
        *reinterpret_cast<__nv_bfloat162*>(out_plane + r * T::row_bytes + col * 2) =
            __floats2bfloat162_rn(acc[j][0], acc[j][1]);
        *reinterpret_cast<__nv_bfloat162*>(out_plane + (r + 8) * T::row_bytes + col * 2) =
            __floats2bfloat162_rn(acc[j][2], acc[j][3]);
      }
    }
    if (threadIdx.x == 0) {
      tma_store_wait_read<0>();  // the previous tile's store has read the output box
    }
    __syncthreads();
    planes_to_box<TW>(planes, out_box, T::row_bytes, T::plane_bytes);
    fence_async_shared();
    __syncthreads();
    if (threadIdx.x == 0) {
      int n, oh0, ow0;
      tile_origin<TW>(t, tiles_w, tiles_h, n, oh0, ow0);
      fence_async_shared();
      tma_store(&y_map, c0, ow0, oh0, n, shared_address(out_box));
      tma_store_commit();
    }
  }
  if (threadIdx.x == 0) {
    tma_store_wait<0>();
  }
}

// Per-block weight and bias gradient partials of x and grad, [blockIdx.x][channel][K * K + 1] fp32, the last entry the
// bias gradient.
template <int TW>
__global__ void __launch_bounds__(kThreads, kBlocksPerSm)
    weight_grad_kernel(const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap g_map,
                       float* __restrict__ partial, int batch, int height, int width, int channels) {
  using T = Tile<TW>;
  constexpr int kTilesN = TW / 8;
  extern __shared__ __align__(128) char shared[];
  char* x_box = shared;
  char* x_planes = x_box + T::box_bytes;
  char* g_box = x_planes + T::planes_bytes;
  char* g_planes = g_box + T::out_box_bytes;
  const unsigned bar = shared_address(g_planes + T::out_planes_bytes);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int c0 = blockIdx.y * kChannels;

  const int tiles_w = (width + TW - 1) / TW;
  const int tiles_h = (height + kRows - 1) / kRows;
  const int64_t tiles = static_cast<int64_t>(batch) * tiles_h * tiles_w;
  auto issue = [&](int64_t t) {
    int n, oh0, ow0;
    tile_origin<TW>(t, tiles_w, tiles_h, n, oh0, ow0);
    mbarrier_expect(bar, T::box_bytes + T::out_box_bytes);
    tma_load(shared_address(x_box), &x_map, c0, ow0 - 4, oh0 - P, n, bar);
    tma_load(shared_address(g_box), &g_map, c0, ow0, oh0, n, bar);
  };

  // ldmatrix.trans addresses. A = x^T (16 columns x 16 rows): matrix q = lane / 8 holds rows (lane % 8) + 8 (q / 2)
  // of columns 8 (q % 2); the x2 refill of a window's upper 8 columns takes rows (lane % 8) + 8 (q % 2). B = g (16
  // rows x 8 columns): rows (lane % 8) + 8 (q % 2).
  const int q = lane / 8;
  const unsigned a_offset = ((lane % 8) + 8 * (q / 2)) * T::row_bytes + 16 * (q % 2);
  const unsigned a_refill_offset = ((lane % 8) + 8 * (q % 2)) * T::row_bytes;
  const unsigned b_offset = ((lane % 8) + 8 * (q % 2)) * T::out_row_bytes;

  float acc[kPerWarp][K][4] = {};
  float bias_acc[kPerWarp] = {};

  int64_t t = blockIdx.x;
  if (threadIdx.x == 0) {
    mbarrier_init(bar, 1);
    fence_mbarrier_init();
    if (t < tiles) {
      issue(t);
    }
  }
  __syncthreads();
  for (unsigned parity = 0; t < tiles; t += gridDim.x, parity ^= 1) {
    mbarrier_wait(bar, parity);
    box_to_planes(x_box, x_planes, T::pixels, T::cols, T::row_bytes, T::plane_bytes);
    box_to_planes(g_box, g_planes, T::out_pixels, TW, T::out_row_bytes, T::out_plane_bytes);
    __syncthreads();
    if (threadIdx.x == 0 && t + gridDim.x < tiles) {
      issue(t + gridDim.x);
    }
#pragma unroll
    for (int p = 0; p < kPerWarp; ++p) {
      const int plane = warp + kWarps * p;
      const unsigned x_plane = shared_address(x_planes + plane * T::plane_bytes);
      const unsigned g_plane = shared_address(g_planes + plane * T::out_plane_bytes);
      unsigned bg[kTilesN][2];
#pragma unroll
      for (int j = 0; j < kTilesN; ++j) {
        ldmatrix_x2_trans(bg[j][0], bg[j][1], g_plane + b_offset + j * 16);
        // Every gradient element of the tile is in exactly one lane's B fragments.
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          bias_acc[p] += __uint_as_float(bg[j][h] << 16) + __uint_as_float(bg[j][h] & 0xffff0000u);
        }
      }
#pragma unroll
      for (int kh = 0; kh < K; ++kh) {
        const unsigned rows = x_plane + kh * T::row_bytes;
        unsigned a[4];
        ldmatrix_x4_trans(a, rows + a_offset);
        mma(acc[p][kh], a, bg[0][0], bg[0][1]);
#pragma unroll
        for (int j = 1; j < kTilesN; ++j) {
          // Window j, columns 8j..8j+15: the previous window's upper half (a1, a3) plus 8 new columns.
          a[0] = a[1];
          a[2] = a[3];
          ldmatrix_x2_trans(a[1], a[3], rows + a_refill_offset + (j + 1) * 16);
          mma(acc[p][kh], a, bg[j][0], bg[j][1]);
        }
      }
    }
    __syncthreads();  // planes free for the next tile
  }

  // Lane entry (i = lane / 4 + 8 (e / 2), j = 2 (lane % 4) + e % 2) of D_kh belongs to tap (kh, i - j - 1); the warp
  // sums each tap with a fixed butterfly.
#pragma unroll
  for (int p = 0; p < kPerWarp; ++p) {
    const int channel = c0 + warp + kWarps * p;
    float* out = partial + (static_cast<int64_t>(blockIdx.x) * channels + channel) * (K * K + 1);
#pragma unroll
    for (int kh = 0; kh < K; ++kh) {
#pragma unroll
      for (int kw = 0; kw < K; ++kw) {
        float v = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          if (lane / 4 + 8 * (e / 2) - (2 * (lane % 4) + e % 2) - 1 == kw) {
            v += acc[p][kh][e];
          }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
          v += __shfl_xor_sync(0xffffffffu, v, offset);
        }
        if (lane == 0) {
          out[kh * K + kw] = v;
        }
      }
    }
    float v = bias_acc[p];
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    if (lane == 0) {
      out[K * K] = v;
    }
  }
}

// Sums the blocks' partials in block order into bf16 grad_weight [C, 1, 7, 7] and grad_bias [C].
__global__ void reduce_partials_kernel(const float* __restrict__ partial, __nv_bfloat16* __restrict__ grad_weight,
                                       __nv_bfloat16* __restrict__ grad_bias, int blocks, int channels) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;  // channel * (K * K + 1) + tap
  if (index >= channels * (K * K + 1)) {
    return;
  }
  float sum = 0.0f;
  for (int b = 0; b < blocks; ++b) {
    sum += partial[static_cast<int64_t>(b) * channels * (K * K + 1) + index];
  }
  const int channel = index / (K * K + 1);
  const int tap = index % (K * K + 1);
  if (tap < K * K) {
    grad_weight[channel * K * K + tap] = __float2bfloat16_rn(sum);
  } else {
    grad_bias[channel] = __float2bfloat16_rn(sum);
  }
}

// A channels-last [N, C, H, W] bf16 tensor as a map over (C, W, H, N) with boxes of kChannels x box_w x box_h x 1;
// reads outside the tensor return zeros.
CUtensorMap channels_last_map(const at::Tensor& t, int box_w, int box_h) {
  const auto n = static_cast<cuuint64_t>(t.size(0));
  const auto c = static_cast<cuuint64_t>(t.size(1));
  const auto h = static_cast<cuuint64_t>(t.size(2));
  const auto w = static_cast<cuuint64_t>(t.size(3));
  const cuuint64_t dims[4] = {c, w, h, n};
  const cuuint64_t strides[3] = {c * 2, w * c * 2, h * w * c * 2};
  const cuuint32_t box[4] = {kChannels, static_cast<cuuint32_t>(box_w), static_cast<cuuint32_t>(box_h), 1};
  return encode_tiled(t, 4, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                      "depthwise_conv");
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

// Blocks along x for one channel chunk: enough to fill every SM kBlocksPerSm times over all chunks.
int64_t blocks_per_chunk(const at::Tensor& x, int64_t tiles) {
  const int64_t resident =
      static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * kBlocksPerSm;
  const int64_t chunks = x.size(1) / kChannels;
  return std::max<int64_t>(1, std::min<int64_t>(tiles, resident / chunks));
}

int64_t tile_count(const at::Tensor& x, int tw) {
  return x.size(0) * ((x.size(2) + kRows - 1) / kRows) * ((x.size(3) + tw - 1) / tw);
}

// Launches conv_kernel with TW-column output tiles.
template <int TW, bool FLIP>
void launch_conv(const at::Tensor& x, const at::Tensor& weight, const at::Tensor* bias, at::Tensor& y) {
  using T = Tile<TW>;
  allow_shared_memory<conv_kernel<TW, FLIP>>(T::conv_bytes);
  const CUtensorMap x_map = channels_last_map(x, T::cols, kInRows);
  const CUtensorMap y_map = channels_last_map(y, TW, kRows);
  const dim3 grid(static_cast<unsigned>(blocks_per_chunk(x, tile_count(x, TW))), x.size(1) / kChannels);
  conv_kernel<TW, FLIP><<<grid, kThreads, T::conv_bytes, at::cuda::getCurrentCUDAStream()>>>(
      x_map, y_map, reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
      bias != nullptr ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr<at::BFloat16>()) : nullptr,
      static_cast<int>(x.size(0)), static_cast<int>(x.size(2)), static_cast<int>(x.size(3)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launches weight_grad_kernel and reduce_partials_kernel with TW-column tiles.
template <int TW>
void launch_weight_grad(const at::Tensor& x, const at::Tensor& grad, at::Tensor& grad_weight, at::Tensor& grad_bias) {
  using T = Tile<TW>;
  allow_shared_memory<weight_grad_kernel<TW>>(T::grad_bytes);
  const CUtensorMap x_map = channels_last_map(x, T::cols, kInRows);
  const CUtensorMap g_map = channels_last_map(grad, TW, kRows);
  const int64_t blocks = blocks_per_chunk(x, tile_count(x, TW));
  const int channels = static_cast<int>(x.size(1));
  auto partial = at::empty({blocks, channels, K * K + 1}, x.options().dtype(at::kFloat));
  const dim3 grid(static_cast<unsigned>(blocks), channels / kChannels);
  weight_grad_kernel<TW><<<grid, kThreads, T::grad_bytes, at::cuda::getCurrentCUDAStream()>>>(
      x_map, g_map, partial.data_ptr<float>(), static_cast<int>(x.size(0)), static_cast<int>(x.size(2)),
      static_cast<int>(x.size(3)), channels);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int total = channels * (K * K + 1);
  reduce_partials_kernel<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(grad_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(grad_bias.data_ptr<at::BFloat16>()), static_cast<int>(blocks), channels);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Output tiles 16 columns wide for maps of at most 16 columns (ConvNeXt's last stage), else 32.
constexpr int kNarrow = 16;
constexpr int kWide = 32;

}  // namespace

bool supported(const at::Tensor& x, const at::Tensor& weight) {
  if (x.scalar_type() != at::kBFloat16 || weight.scalar_type() != at::kBFloat16 || weight.size(2) != K ||
      weight.size(3) != K || x.size(1) % kChannels != 0 || x.numel() == 0) {
    return false;
  }
  if (!x.is_contiguous(at::MemoryFormat::ChannelsLast) || reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 != 0) {
    return false;
  }
  return at::cuda::getCurrentDeviceProperties()->major >= 9;
}

at::Tensor conv(const at::Tensor& x, const at::Tensor& weight, const at::Tensor* bias, bool flip) {
  auto y = at::empty_like(x, x.options(), at::MemoryFormat::ChannelsLast);
  const auto weight_c = weight.contiguous();
  const bool narrow = x.size(3) <= kNarrow;
  if (flip) {
    narrow ? launch_conv<kNarrow, true>(x, weight_c, bias, y) : launch_conv<kWide, true>(x, weight_c, bias, y);
  } else {
    narrow ? launch_conv<kNarrow, false>(x, weight_c, bias, y) : launch_conv<kWide, false>(x, weight_c, bias, y);
  }
  return y;
}

std::tuple<at::Tensor, at::Tensor> weight_grad(const at::Tensor& x, const at::Tensor& grad, const at::Tensor& weight) {
  TORCH_CHECK(reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0, "depthwise_conv: grad must be 16-byte aligned");
  auto grad_weight = at::empty_like(weight, at::MemoryFormat::Contiguous);
  auto grad_bias = at::empty({x.size(1)}, weight.options());
  if (x.size(3) <= kNarrow) {
    launch_weight_grad<kNarrow>(x, grad, grad_weight, grad_bias);
  } else {
    launch_weight_grad<kWide>(x, grad, grad_weight, grad_bias);
  }
  return {grad_weight, grad_bias};
}

}  // namespace depthwise_tc
}  // namespace py123d_garage_cuda
