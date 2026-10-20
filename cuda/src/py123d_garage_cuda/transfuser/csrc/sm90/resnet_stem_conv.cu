// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// ResNet's stem convolution, 7 x 7, stride 2, padding 3, no bias, 2 or 3 to 64 channels, on channels-last bf16 tensors
// with fp32 accumulation; the stem's input is data, so there is no input gradient.
//
// The input arrives padded to 4 channels (8 bytes a pixel) and each kernel row's 7 taps pad to 8, so the im2col matrix
// is [pixels, 7 x 8 x 4] and every 8 of its columns, two taps of one pixel, are one 16-byte run that ldmatrix reads.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include <cstdint>

#include "transfuser/csrc/sm80/ptx.cuh"
#include "transfuser/csrc/sm90/tma.cuh"

namespace py123d_garage_cuda {
namespace {

constexpr int kOut = 64;
constexpr int kTaps = 7;
// Input channels as stored, and taps per kernel row as computed (the eighth has zero weights).
constexpr int kChannels = 4;
constexpr int kRowTaps = 8;
// im2col columns per kernel row and in all; the GEMM's K.
constexpr int kRowK = kRowTaps * kChannels;
constexpr int kK = kTaps * kRowK;
constexpr int kKSteps = kK / 16;
// An output tile is kTile x kTile pixels; its input patch kPatchRows x kPatchCols pixels of 8 bytes. The patch row
// stride, 304 bytes, keeps ldmatrix rows 16-byte aligned.
constexpr int kTile = 16;
constexpr int kPatchRows = 2 * (kTile - 1) + kTaps;
constexpr int kPatchCols = 2 * (kTile - 1) + kRowTaps;
constexpr int kPatchElems = kPatchRows * kPatchCols * kChannels;
// Weight row stride in bf16: 464 bytes, an odd number of 16-byte units, for conflict-free ldmatrix.
constexpr int kWStride = kK + 8;
// Row stride in bf16 of a staged 64-channel pixel: 144 bytes, an odd number of 16-byte units.
constexpr int kStage = 72;
constexpr int kBlocksPerSm = 2;
constexpr int kForwardWarps = 8;
// The weight gradient's warp w sums kernel row w.
constexpr int kGradWarps = kTaps;

struct TileGrid {
  int height, width, out_height, out_width, tiles_h, tiles_w, num_tiles;

  __device__ __forceinline__ void locate(int tile, int& n, int& oh0, int& ow0) const {
    const int per_image = tiles_h * tiles_w;
    n = tile / per_image;
    const int rest = tile - n * per_image;
    const int th = rest / tiles_w;
    oh0 = th * kTile;
    ow0 = (rest - th * tiles_w) * kTile;
  }
};

// cp.async of the input patch of the tile at output (oh0, ow0): pixels from row 2 oh0 - 3 and column 2 ow0 - 3 on,
// zero outside the image.
template <int kThreads>
__device__ __forceinline__ void load_patch(const __nv_bfloat16* x, __nv_bfloat16* patch, const TileGrid& grid, int n,
                                           int oh0, int ow0) {
  const int ih0 = 2 * oh0 - 3;
  const int iw0 = 2 * ow0 - 3;
  for (int i = threadIdx.x; i < kPatchRows * kPatchCols; i += kThreads) {
    const int row = i / kPatchCols;
    const int col = i - row * kPatchCols;
    const int ih = ih0 + row;
    const int iw = iw0 + col;
    const bool valid = ih >= 0 && ih < grid.height && iw >= 0 && iw < grid.width;
    const __nv_bfloat16* src =
        valid ? x + ((static_cast<int64_t>(n) * grid.height + ih) * grid.width + iw) * kChannels : x;
    cp_async_8(patch + i * kChannels, src, valid);
  }
}

// The weight as [64, kK] rows in (kh, kw, ci) order with zeros for the eighth tap and the padding channels.
template <int C, int kThreads>
__device__ __forceinline__ void stage_weight(const __nv_bfloat16* weight, __nv_bfloat16* w_s) {
  for (int i = threadIdx.x; i < kOut * kK; i += kThreads) {
    const int co = i / kK;
    const int k = i - co * kK;
    const int kh = k / kRowK;
    const int kw = (k / kChannels) % kRowTaps;
    const int ci = k % kChannels;
    w_s[co * kWStride + k] =
        kw < kTaps && ci < C ? weight[((co * C + ci) * kTaps + kh) * kTaps + kw] : __float2bfloat16(0.0f);
  }
}

// y = conv(x, weight) and per block the [3, 64] count, mean and centred sum of squares of its bf16 outputs. A
// persistent block runs 16 x 16 output tiles; warp w computes tile rows 2w and 2w + 1 (one m16 tile each).
template <int C>
__global__ void __launch_bounds__(32 * kForwardWarps, kBlocksPerSm)
    stem_conv_forward_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
                             __nv_bfloat16* __restrict__ y, float* __restrict__ partial, TileGrid grid) {
  constexpr int kThreads = 32 * kForwardWarps;
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* w_s = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* patches = w_s + kOut * kWStride;
  __nv_bfloat16* stage_all = patches + 2 * kPatchElems;
  float* warp_stats = reinterpret_cast<float*>(stage_all + kForwardWarps * 2 * kTile * kStage);

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int t = lane % 4;
  __nv_bfloat16* stage = stage_all + warp * 2 * kTile * kStage;

  int tile = blockIdx.x;
  if (tile < grid.num_tiles) {
    int n, oh0, ow0;
    grid.locate(tile, n, oh0, ow0);
    load_patch<kThreads>(x, patches, grid, n, oh0, ow0);
  }
  cp_async_commit();
  stage_weight<C, kThreads>(weight, w_s);

  // ldmatrix row addresses of this lane. A: matrix q = lane / 8 holds pixels 8 (q % 2) + [0, 8) and the k half q / 2
  // of a k step, i.e. taps 2 (q / 2) + {0, 1} of the step's four. B: rows (output channels) 8 (lane / 16) + [0, 8),
  // k half (lane / 8) % 2.
  const int a_pixel = (lane / 8 % 2) * 8 + lane % 8;
  const int a_offset = (2 * a_pixel + 2 * (lane / 16)) * kChannels;
  const int b_offset = ((lane / 16) * 8 + lane % 8) * kWStride + (lane / 8 % 2) * 8;

  // Per lane, channels 2 lane and 2 lane + 1: count, shift (a value seen first), and the shifted sums.
  int count = 0;
  float shift0 = 0.0f, shift1 = 0.0f, sum0 = 0.0f, sum1 = 0.0f, squares0 = 0.0f, squares1 = 0.0f;

  int buffer = 0;
  for (; tile < grid.num_tiles; tile += gridDim.x) {
    const int next = tile + gridDim.x;
    if (next < grid.num_tiles) {
      int n, oh0, ow0;
      grid.locate(next, n, oh0, ow0);
      load_patch<kThreads>(x, patches + (buffer ^ 1) * kPatchElems, grid, n, oh0, ow0);
    }
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();

    const __nv_bfloat16* patch = patches + buffer * kPatchElems;
    float acc[2][8][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[m][j][q] = 0.0f;

#pragma unroll
    for (int ks = 0; ks < kKSteps; ++ks) {
      // Step ks covers kernel row ks / 2, taps 4 (ks % 2) + [0, 4).
      const int kh = ks / 2;
      const int tap0 = 4 * (ks % 2);
      uint32_t a[2][4];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int row = 2 * (2 * warp + m) + kh;
        ldmatrix_x4(a[m], patch + row * kPatchCols * kChannels + tap0 * kChannels + a_offset);
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b[4];
        ldmatrix_x4(b, w_s + 16 * np * kWStride + 16 * ks + b_offset);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          mma(acc[m][2 * np], a[m], b[0], b[1]);
          mma(acc[m][2 * np + 1], a[m], b[2], b[3]);
        }
      }
    }

    // Stage the bf16 outputs as [32 pixels, 64 channels] once the bulk stores of the previous tile have read the
    // stage, then statistics and one 128-byte bulk store per pixel from the stage.
    tma_store_wait_read<0>();
    __syncwarp();
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        uint32_t* row0 = reinterpret_cast<uint32_t*>(stage + (16 * m + g) * kStage + 8 * j + 2 * t);
        uint32_t* row8 = reinterpret_cast<uint32_t*>(stage + (16 * m + g + 8) * kStage + 8 * j + 2 * t);
        *row0 = pack_bf16(acc[m][j][0], acc[m][j][1]);
        *row8 = pack_bf16(acc[m][j][2], acc[m][j][3]);
      }
    }
    int n, oh0, ow0;
    grid.locate(tile, n, oh0, ow0);
    const int oh = oh0 + 2 * warp + lane / kTile;
    const int ow = ow0 + lane % kTile;
    const bool inside = oh < grid.out_height && ow < grid.out_width;
    // Lane l stores pixel l; in edge tiles the pixels outside the output become zeros for the statistics below. The
    // zeroing overwrites rows the other lanes staged above, so it waits for them.
    const int rows = max(0, min(2, grid.out_height - oh0 - 2 * warp));
    const int outside = 2 * kTile - rows * min(kTile, grid.out_width - ow0);
    __syncwarp();
    if (outside > 0 && !inside) {
#pragma unroll
      for (int part = 0; part < 8; ++part) *reinterpret_cast<int4*>(stage + lane * kStage + 8 * part) = int4{};
    }
    fence_async_shared();
    __syncwarp();
    if (inside) {
      bulk_store_128(y + ((static_cast<int64_t>(n) * grid.out_height + oh) * grid.out_width + ow) * kOut,
                     stage + lane * kStage);
    }
    tma_store_commit();

    // Sum all 32 pixels and take the shifted zeros of the outside ones back out.
    if (count == 0 && rows > 0) {
      const uint32_t first = *reinterpret_cast<const uint32_t*>(stage + 2 * lane);
      shift0 = __uint_as_float(first << 16);
      shift1 = __uint_as_float(first & 0xffff0000u);
    }
    count += 2 * kTile - outside;
    float tile_sum0 = 0.0f, tile_sum1 = 0.0f, tile_squares0 = 0.0f, tile_squares1 = 0.0f;
#pragma unroll
    for (int pixel = 0; pixel < 2 * kTile; ++pixel) {
      const uint32_t pair = *reinterpret_cast<const uint32_t*>(stage + pixel * kStage + 2 * lane);
      const float d0 = __uint_as_float(pair << 16) - shift0;
      const float d1 = __uint_as_float(pair & 0xffff0000u) - shift1;
      tile_sum0 += d0;
      tile_sum1 += d1;
      tile_squares0 = fmaf(d0, d0, tile_squares0);
      tile_squares1 = fmaf(d1, d1, tile_squares1);
    }
    sum0 += tile_sum0 + outside * shift0;
    sum1 += tile_sum1 + outside * shift1;
    squares0 += tile_squares0 - outside * shift0 * shift0;
    squares1 += tile_squares1 - outside * shift1 * shift1;
    __syncthreads();
    buffer ^= 1;
  }
  cp_async_wait<0>();
  tma_store_wait<0>();

  // Per warp and channel: count, mean, centred sum of squares; then merged over the warps in order.
  float* mine = warp_stats + warp * 3 * kOut;
#pragma unroll
  for (int e = 0; e < 2; ++e) {
    const int c = 2 * lane + e;
    const float sum = e == 0 ? sum0 : sum1;
    const float squares = e == 0 ? squares0 : squares1;
    const float shift = e == 0 ? shift0 : shift1;
    const float mean = count > 0 ? sum / count : 0.0f;
    mine[c] = static_cast<float>(count);
    mine[kOut + c] = shift + mean;
    mine[2 * kOut + c] = count > 0 ? fmaxf(squares - sum * mean, 0.0f) : 0.0f;
  }
  __syncthreads();
  if (threadIdx.x < kOut) {
    const int c = threadIdx.x;
    float total = 0.0f, mean = 0.0f, m2 = 0.0f;
    for (int w = 0; w < kForwardWarps; ++w) {
      const float* stats = warp_stats + w * 3 * kOut;
      const float other = stats[c];
      if (other == 0.0f) continue;
      const float merged = total + other;
      const float delta = stats[kOut + c] - mean;
      mean += delta * (other / merged);
      m2 += stats[2 * kOut + c] + delta * delta * (total * other / merged);
      total = merged;
    }
    float* out = partial + blockIdx.x * 3 * kOut;
    out[c] = total;
    out[kOut + c] = mean;
    out[2 * kOut + c] = m2;
  }
}

// Per-block weight gradient partials [block, kK, 64] fp32. A persistent block runs output tiles; warp w owns kernel row
// w, im2col columns 32 w + [0, 32) (two m16 tiles), for all 64 output channels, summing 16 pixels per mma.
__global__ void __launch_bounds__(32 * kGradWarps, kBlocksPerSm)
    stem_conv_weight_grad_kernel(const __nv_bfloat16* __restrict__ grad, const __nv_bfloat16* __restrict__ x,
                                 float* __restrict__ partial, TileGrid grid) {
  constexpr int kThreads = 32 * kGradWarps;
  constexpr int kStageElems = kTile * kTile * kStage;
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* stages = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* patches = stages + 2 * kStageElems;

  const int kh = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int t = lane % 4;

  auto load = [&](int tile_index, int which) {
    int n, oh0, ow0;
    grid.locate(tile_index, n, oh0, ow0);
    load_patch<kThreads>(x, patches + which * kPatchElems, grid, n, oh0, ow0);
    __nv_bfloat16* stage = stages + which * kStageElems;
    for (int i = threadIdx.x; i < kTile * kTile * 8; i += kThreads) {
      const int pixel = i / 8;
      const int part = i % 8;
      const int oh = oh0 + pixel / kTile;
      const int ow = ow0 + pixel % kTile;
      const bool valid = oh < grid.out_height && ow < grid.out_width;
      const __nv_bfloat16* src =
          valid ? grad + ((static_cast<int64_t>(n) * grid.out_height + oh) * grid.out_width + ow) * kOut + 8 * part
                : grad;
      cp_async_16(stage + pixel * kStage + 8 * part, src, valid);
    }
  };

  // ldmatrix.trans row addresses of this lane. A (im2col^T, k x pixels): matrix q = lane / 8 holds k half q % 2 (taps
  // 2 (q % 2) + {0, 1} of the m tile's four) and pixels 8 (q / 2) + [0, 8). B (gradient, pixels x channels): pixels
  // 8 (q % 2) + [0, 8), channels 8 (q / 2) + [0, 8).
  const int a_pixel = (lane / 16) * 8 + lane % 8;
  const int a_offset = kh * kPatchCols * kChannels + (2 * a_pixel + 2 * (lane / 8 % 2)) * kChannels;
  const int b_offset = ((lane / 8 % 2) * 8 + lane % 8) * kStage + (lane / 16) * 8;

  float acc[2][8][4];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[m][j][q] = 0.0f;

  int tile = blockIdx.x;
  if (tile < grid.num_tiles) load(tile, 0);
  cp_async_commit();
  int buffer = 0;
  for (; tile < grid.num_tiles; tile += gridDim.x) {
    const int next = tile + gridDim.x;
    if (next < grid.num_tiles) load(next, buffer ^ 1);
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();

    const __nv_bfloat16* stage = stages + buffer * kStageElems;
    const __nv_bfloat16* patch = patches + buffer * kPatchElems;
#pragma unroll 2
    for (int r = 0; r < kTile; ++r) {
      uint32_t a[2][4];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        ldmatrix_x4_trans(a[m], patch + 2 * r * kPatchCols * kChannels + 4 * m * kChannels + a_offset);
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b[4];
        ldmatrix_x4_trans(b, stage + r * kTile * kStage + 16 * np + b_offset);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          mma(acc[m][2 * np], a[m], b[0], b[1]);
          mma(acc[m][2 * np + 1], a[m], b[2], b[3]);
        }
      }
    }
    __syncthreads();
    buffer ^= 1;
  }
  cp_async_wait<0>();

  // partial[block, k, co] for the block's pixels.
  float* out = partial + static_cast<int64_t>(blockIdx.x) * kK * kOut;
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int k = kh * kRowK + 16 * m + g;
      const int co = 8 * j + 2 * t;
      *reinterpret_cast<float2*>(out + k * kOut + co) = make_float2(acc[m][j][0], acc[m][j][1]);
      *reinterpret_cast<float2*>(out + (k + 8) * kOut + co) = make_float2(acc[m][j][2], acc[m][j][3]);
    }
  }
}

// grad_weight[co, ci, kh, kw] = sum over blocks, in block order, of partial[block, (kh, kw, ci), co].
template <int C>
__global__ void stem_conv_weight_grad_reduce_kernel(const float* __restrict__ partial,
                                                    __nv_bfloat16* __restrict__ grad_weight, int blocks) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kOut * C * kTaps * kTaps) return;
  // Consecutive threads take consecutive output channels, so the partial reads coalesce.
  const int co = index % kOut;
  const int tap = index / kOut;
  const int ci = tap / (kTaps * kTaps);
  const int kh = tap / kTaps % kTaps;
  const int kw = tap % kTaps;
  const int k = kh * kRowK + kw * kChannels + ci;
  float total = 0.0f;
  for (int b = 0; b < blocks; ++b) total += partial[(static_cast<int64_t>(b) * kK + k) * kOut + co];
  grad_weight[((co * C + ci) * kTaps + kh) * kTaps + kw] = __float2bfloat16(total);
}

constexpr int kFinalizeThreads = 256;

struct Moments {
  float count, mean, m2;
};

// Chan et al.'s merge of two sets' count, mean and centred sum of squares.
__device__ __forceinline__ Moments merge(const Moments& a, const Moments& b) {
  if (b.count == 0.0f) return a;
  if (a.count == 0.0f) return b;
  const float count = a.count + b.count;
  const float delta = b.mean - a.mean;
  return {count, a.mean + delta * (b.count / count), a.m2 + b.m2 + delta * delta * (a.count * b.count / count)};
}

// Merges the blocks' (count, mean, centred sum of squares) of channel blockIdx.x, in a fixed order (strided per
// thread, then a tree), into values[5, C] = mean, biased variance, rstd, scale = weight * rstd and
// shift = bias - mean * scale.
__global__ void __launch_bounds__(kFinalizeThreads)
    batch_norm_finalize_kernel(const float* __restrict__ partial, const float* __restrict__ weight,
                               const float* __restrict__ bias, float* __restrict__ values, int blocks, int channels,
                               float eps) {
  __shared__ Moments shared[kFinalizeThreads];
  const int c = blockIdx.x;
  Moments moments{0.0f, 0.0f, 0.0f};
  for (int b = threadIdx.x; b < blocks; b += kFinalizeThreads) {
    const float* stats = partial + static_cast<int64_t>(b) * 3 * channels;
    moments = merge(moments, {stats[c], stats[channels + c], stats[2 * channels + c]});
  }
  shared[threadIdx.x] = moments;
  __syncthreads();
  for (int stride = kFinalizeThreads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) shared[threadIdx.x] = merge(shared[threadIdx.x], shared[threadIdx.x + stride]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const Moments all = shared[0];
    const float variance = all.m2 / all.count;
    const float rstd = rsqrtf(variance + eps);
    const float scale = weight[c] * rstd;
    values[c] = all.mean;
    values[channels + c] = variance;
    values[2 * channels + c] = rstd;
    values[3 * channels + c] = scale;
    values[4 * channels + c] = bias[c] - all.mean * scale;
  }
}

TileGrid make_grid(const at::Tensor& x) {
  TileGrid grid;
  grid.height = static_cast<int>(x.size(2));
  grid.width = static_cast<int>(x.size(3));
  grid.out_height = (grid.height - 1) / 2 + 1;
  grid.out_width = (grid.width - 1) / 2 + 1;
  grid.tiles_h = (grid.out_height + kTile - 1) / kTile;
  grid.tiles_w = (grid.out_width + kTile - 1) / kTile;
  grid.num_tiles = static_cast<int>(x.size(0)) * grid.tiles_h * grid.tiles_w;
  return grid;
}

int persistent_blocks(const TileGrid& grid) {
  const int blocks = at::cuda::getCurrentDeviceProperties()->multiProcessorCount * kBlocksPerSm;
  return std::max(1, std::min(blocks, grid.num_tiles));
}

void check_input(const at::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 4, name, ": x must be a 4-D CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, name, ": x must be bf16");
  TORCH_CHECK(x.size(1) == kChannels, name, ": x must be padded to 4 channels");
  TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast), name, ": x must be channels-last");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0, name, ": x must be 8-byte aligned");
  TORCH_CHECK(x.numel() < (int64_t{1} << 31), name, ": x is too large");
}

constexpr size_t kForwardSmem =
    (kOut * kWStride + 2 * kPatchElems + kForwardWarps * 2 * kTile * kStage) * 2 + kForwardWarps * 3 * kOut * 4;
constexpr size_t kGradSmem = (2 * kTile * kTile * kStage + 2 * kPatchElems) * 2;

// Launches the forward kernel with one block per row of partial.
template <int C>
void launch_forward(const at::Tensor& x, const at::Tensor& weight, at::Tensor& y, at::Tensor& partial, TileGrid grid) {
  auto kernel = stem_conv_forward_kernel<C>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kForwardSmem));
  kernel<<<partial.size(0), 32 * kForwardWarps, kForwardSmem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), partial.data_ptr<float>(), grid);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Forward op: x [N, 4, H, W] channels-last bf16 (channels past the weight's zero), weight [64, 2 or 3, 7, 7] bf16;
// returns the channels-last bf16 output and the per-block statistics [blocks, 3, 64].
std::tuple<at::Tensor, at::Tensor> resnet_stem_conv(const at::Tensor& x, const at::Tensor& weight) {
  check_input(x, "resnet_stem_conv");
  TORCH_CHECK(weight.dim() == 4 && weight.size(0) == kOut && (weight.size(1) == 2 || weight.size(1) == 3) &&
                  weight.size(2) == kTaps && weight.size(3) == kTaps,
              "resnet_stem_conv: weight must be [64, 2 or 3, 7, 7]");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "resnet_stem_conv: weight must be bf16");
  const c10::cuda::CUDAGuard guard(x.device());
  const TileGrid grid = make_grid(x);
  auto y = at::empty({x.size(0), kOut, grid.out_height, grid.out_width},
                     x.options().memory_format(at::MemoryFormat::ChannelsLast));
  auto partial = at::empty({persistent_blocks(grid), 3, kOut}, x.options().dtype(at::kFloat));
  const auto w = weight.contiguous();
  if (weight.size(1) == 3) {
    launch_forward<3>(x, w, y, partial, grid);
  } else {
    launch_forward<2>(x, w, y, partial, grid);
  }
  return {y, partial};
}

// The bf16 weight gradient [64, channels, 7, 7] from the output gradient grad [N, 64, out_h, out_w] and the input x.
at::Tensor resnet_stem_conv_weight_grad(const at::Tensor& grad, const at::Tensor& x, int64_t channels) {
  check_input(x, "resnet_stem_conv_weight_grad");
  TORCH_CHECK(channels == 2 || channels == 3, "resnet_stem_conv_weight_grad: channels must be 2 or 3");
  const TileGrid grid = make_grid(x);
  TORCH_CHECK(grad.dim() == 4 && grad.size(0) == x.size(0) && grad.size(1) == kOut && grad.size(2) == grid.out_height &&
                  grad.size(3) == grid.out_width,
              "resnet_stem_conv_weight_grad: grad must be [N, 64, out_h, out_w]");
  TORCH_CHECK(grad.scalar_type() == at::kBFloat16, "resnet_stem_conv_weight_grad: grad must be bf16");
  const c10::cuda::CUDAGuard guard(x.device());
  const auto g = grad.contiguous(at::MemoryFormat::ChannelsLast);
  auto grad_weight = at::empty({kOut, channels, kTaps, kTaps}, x.options());
  const int blocks = persistent_blocks(grid);
  auto partial = at::empty({blocks, kK, kOut}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(stem_conv_weight_grad_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kGradSmem));
  stem_conv_weight_grad_kernel<<<blocks, 32 * kGradWarps, kGradSmem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      partial.data_ptr<float>(), grid);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int outputs = kOut * static_cast<int>(channels) * kTaps * kTaps;
  auto out = reinterpret_cast<__nv_bfloat16*>(grad_weight.data_ptr());
  if (channels == 3) {
    stem_conv_weight_grad_reduce_kernel<3>
        <<<(outputs + 255) / 256, 256, 0, stream>>>(partial.data_ptr<float>(), out, blocks);
  } else {
    stem_conv_weight_grad_reduce_kernel<2>
        <<<(outputs + 255) / 256, 256, 0, stream>>>(partial.data_ptr<float>(), out, blocks);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_weight;
}

// Merges partial [blocks, 3, C] into values [5, C]: mean, biased variance, rstd, scale and shift.
at::Tensor batch_norm_finalize(const at::Tensor& partial, const at::Tensor& weight, const at::Tensor& bias,
                               double eps) {
  TORCH_CHECK(partial.is_cuda() && partial.dim() == 3 && partial.size(1) == 3,
              "batch_norm_finalize: partial must be [B, 3, C]");
  TORCH_CHECK(partial.scalar_type() == at::kFloat && partial.is_contiguous(),
              "batch_norm_finalize: partial must be contiguous fp32");
  const int channels = static_cast<int>(partial.size(2));
  TORCH_CHECK(weight.numel() == channels && bias.numel() == channels,
              "batch_norm_finalize: weight and bias must have C values");
  const c10::cuda::CUDAGuard guard(partial.device());
  const auto w = weight.to(at::kFloat).contiguous();
  const auto b = bias.to(at::kFloat).contiguous();
  auto values = at::empty({5, channels}, partial.options());
  batch_norm_finalize_kernel<<<channels, kFinalizeThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      partial.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), values.data_ptr<float>(),
      static_cast<int>(partial.size(0)), channels, static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return values;
}

}  // namespace

TORCH_LIBRARY_IMPL(py123d_garage_cuda, CUDA, m) {
  m.impl("resnet_stem_conv", &resnet_stem_conv);
  m.impl("resnet_stem_conv_weight_grad", &resnet_stem_conv_weight_grad);
  m.impl("batch_norm_finalize", &batch_norm_finalize);
}

}  // namespace py123d_garage_cuda
