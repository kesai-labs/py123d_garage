// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// RegNet's grouped 3 x 3 convolution (24 channels per group, padding 1, stride 1 or 2) on channels-last bf16 tensors,
// accumulated in fp32 on tensor cores (m16n8k16 + m16n8k8, so the 24-wide groups waste no lanes); sm_90 only (TMA).
// The convolution sums tap by tap, channels 0..15 then 16..23, in cuDNN's order, so the forward matches it bit for bit.
//
// Every kernel works on three groups (72 channels, one 144-byte run per pixel) of tiles of 8-pixel-wide rows, with
// persistent blocks and one thread issuing the TMA loads a few tiles ahead. Partial sums are reduced in a fixed order.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include <algorithm>
#include <initializer_list>
#include <map>
#include <mutex>
#include <tuple>

#include "transfuser/csrc/sm80/ptx.cuh"
#include "transfuser/csrc/sm90/tma.cuh"

namespace py123d_garage_cuda {
namespace {

constexpr int kGroup = 24;
constexpr int kGroupsPerBlock = 3;
constexpr int kBlockChannels = kGroup * kGroupsPerBlock;
constexpr int kPixelBytes = kBlockChannels * 2;
constexpr int kTaps = 9;
// Rows of the weight gradient per group: (tap, input channel), in 8-channel chunks, padded to 14 m16 tiles.
constexpr int kGradRows = kTaps * kGroup;
constexpr int kGradChunks = kGradRows / 8;
constexpr int kGradTiles = (kGradChunks + 1) / 2;
// TMA pipeline depth.
constexpr int kStages = 2;
// Blocks per thread-block cluster of the weight gradient, which adds its blocks' partial sums on chip.
constexpr int kGradCluster = 8;
constexpr int kGradBlockSumBytes = kBlockChannels * kGradRows * 4;

constexpr int round_up(int bytes) { return (bytes + 127) / 128 * 128; }

__device__ __forceinline__ void store_shared(uint32_t address, float first, float second) {
  __nv_bfloat162 pair = __floats2bfloat162_rn(first, second);
  asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(address), "r"(*reinterpret_cast<uint32_t*>(&pair)));
}

__device__ __forceinline__ float to_float(float v) { return v; }

__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ void store(float* p, float v) { *p = v; }

__device__ __forceinline__ void store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

// Element strides of a [C, 24, 3, 3] weight (contiguous or channels-last).
struct WeightStrides {
  int o;
  int i;
  int h;
  int w;

  __device__ __forceinline__ int at(int out, int in, int tap) const {
    return out * o + in * i + (tap / 3) * h + (tap % 3) * w;
  }
};

// Two weights rounded to bf16 like autocast, the first in the low half.
template <typename weight_t>
__device__ __forceinline__ uint32_t weight_pair(const weight_t* weight, int first, int second) {
  __nv_bfloat162 pair = __floats2bfloat162_rn(to_float(weight[first]), to_float(weight[second]));
  return *reinterpret_cast<uint32_t*>(&pair);
}

// The staged source pixels of a tile of TH x TW outputs (MODE 0, 1) or cells (MODE 2), [row][col][72 channels]:
// MODE 0: stride-1 convolution, tile pixel (r, c) reads the staged rows r .. r + 2 and columns c .. c + 2.
// MODE 1: stride-2 convolution, the staged rows 2r .. 2r + 2 and columns 2c .. 2c + 2.
// MODE 2: input gradient of the stride-2 convolution: tile pixel (r, c) is the cell of the four input pixels
//         (2r + a, 2c + b), which read the staged output-gradient rows r .. r + 1 and columns c .. c + 1.
template <int MODE, int TH, int TW>
struct Tile {
  static constexpr int kRows = MODE == 0 ? TH + 2 : MODE == 1 ? 2 * TH + 1 : TH + 1;
  static constexpr int kCols = MODE == 0 ? TW + 2 : MODE == 1 ? 2 * TW + 1 : TW + 1;
  static constexpr int kBytes = kRows * kCols * kPixelBytes;
  static constexpr int kSharedBytes = kBytes;
  static constexpr int kScale = MODE == 1 ? 2 : 1;
  static constexpr int kShift = MODE == 2 ? 0 : 1;
  static constexpr int kPixels = TH * TW;

  // The staged pixel that tile pixel (r, c) reads at tap offset (0, 0).
  __host__ __device__ static constexpr int base(int r, int c) {
    return MODE == 1 ? 2 * (r * kCols + c) : r * kCols + c;
  }
};

// The TMA load of the staged source pixels of tile (tile_h, tile_w) of image n, T::kBytes.
template <int MODE, int TH, int TW>
__device__ __forceinline__ void load_tile(uint32_t destination, const CUtensorMap* map, int c0, int tile_h, int tile_w,
                                          int n, uint32_t barrier) {
  using T = Tile<MODE, TH, TW>;
  const int h0 = T::kScale * tile_h * TH - T::kShift;
  const int w0 = T::kScale * tile_w * TW - T::kShift;
  tma_load(destination, map, c0, w0, h0, n, barrier);
}

// The input-gradient taps of the stride-2 convolution by phase (a, b) = (0, 0), (0, 1), (1, 0), (1, 1), as kh, kw and
// the staged row and column offsets; the functions fold to constants in the unrolled loops.
__host__ __device__ constexpr int phase_tap(int t, int field) {
  constexpr int kh[kTaps] = {1, 1, 1, 0, 2, 0, 0, 2, 2};
  constexpr int kw[kTaps] = {1, 0, 2, 1, 1, 0, 2, 0, 2};
  return field == 0 ? kh[t] : field == 1 ? kw[t] : field == 2 ? (kh[t] == 0) : (kw[t] == 0);
}

__host__ __device__ constexpr int phase_start(int phase) {
  return phase == 0 ? 0 : phase == 1 ? 1 : phase == 2 ? 3 : phase == 3 ? 5 : 9;
}

// The staged pixel offset of tap t.
template <int MODE, int TH, int TW>
__host__ __device__ constexpr int tap_offset(int t) {
  using T = Tile<MODE, TH, TW>;
  if (MODE == 0) {
    return (t / 3) * T::kCols + t % 3;
  }
  if (MODE == 1) {
    return (t / 3) * T::kCols + t % 3;
  }
  return phase_tap(t, 2) * T::kCols + phase_tap(t, 3);
}

// The kernel tap whose weights tap t multiplies.
template <int MODE>
__host__ __device__ constexpr int weight_tap(int t) {
  return MODE == 2 ? phase_tap(t, 0) * 3 + phase_tap(t, 1) : t;
}

// With FUSED, the convolution's source is relu(y * scale + shift) of the batch norm input y, applied to the staged
// tiles; the forward also sums its output and its square per channel, the input gradient the gradients of shift and of
// scale (the latter around `center`, the batch mean, for precision), per block into partial[block][2][C].
struct BnArgs {
  const float* scale;
  const float* shift;
  const float* center;
  float* partial;
  // The source image of the convolution and the weight gradient.
  int height;
  int width;
};

// Threads of the fused pass over the staged pixels: thread t handles the 16-byte vector t % 9 (8 channels) of every
// (blockDim / 9)-th pixel from t / 9, so its channels and their scale and shift stay fixed.
constexpr int kVectors = kPixelBytes / 16;

template <int WPG>
struct VectorLane {
  static constexpr int kStep = 32 * kGroupsPerBlock * WPG / kVectors;
  int vector;
  int first;
  bool active;

  __device__ __forceinline__ VectorLane() {
    vector = threadIdx.x % kVectors;
    first = threadIdx.x / kVectors;
    active = first < kStep;
  }

  __device__ __forceinline__ void sync() const { __syncthreads(); }
};

// Loaded per tile from L1: held across the tile loop, they make the kernels spill registers.
__device__ __forceinline__ void load_channels(const float* values, int c, float (&out)[8]) {
  const float4 low = __ldg(reinterpret_cast<const float4*>(values + c));
  const float4 high = __ldg(reinterpret_cast<const float4*>(values + c + 4));
  out[0] = low.x, out[1] = low.y, out[2] = low.z, out[3] = low.w;
  out[4] = high.x, out[5] = high.y, out[6] = high.z, out[7] = high.w;
}

// relu(first) in the low and relu(second) in the high half, rounded to bf16, in one instruction.
__device__ __forceinline__ uint32_t relu_bf16x2(float first, float second) {
  uint32_t packed;
  asm("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(packed) : "f"(second), "f"(first));
  return packed;
}

__device__ __forceinline__ void bn_relu_vector(uint4* address, const float (&scale)[8], const float (&shift)[8]) {
  uint4 v = *address;
  uint32_t* pairs = reinterpret_cast<uint32_t*>(&v);
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    // bf16 to fp32 is a shift.
    const float low = __uint_as_float(pairs[e] << 16);
    const float high = __uint_as_float(pairs[e] & 0xffff0000u);
    pairs[e] = relu_bf16x2(fmaf(low, scale[2 * e], shift[2 * e]), fmaf(high, scale[2 * e + 1], shift[2 * e + 1]));
  }
  *address = v;
}

template <int ROWS, int COLS, int WPG>
__device__ __forceinline__ void bn_relu_staged_pixels(unsigned char* tile, int h0, int w0, int height, int width,
                                                      const float (&scale)[8], const float (&shift)[8],
                                                      const VectorLane<WPG>& lane) {
  constexpr int kPixels = ROWS * COLS;
  constexpr int kStep = VectorLane<WPG>::kStep;
  if (!lane.active) {
    return;
  }
  uint4* vectors = reinterpret_cast<uint4*>(tile) + lane.first * kVectors + lane.vector;
  if (h0 >= 0 && w0 >= 0 && h0 + ROWS <= height && w0 + COLS <= width) {
#pragma unroll 4
    for (int k = 0; k < (kPixels + kStep - 1) / kStep; ++k) {
      if (lane.first + k * kStep < kPixels) {
        bn_relu_vector(vectors + k * kStep * kVectors, scale, shift);
      }
    }
    return;
  }
  for (int p = lane.first; p < kPixels; p += kStep) {
    const int h = h0 + p / COLS;
    const int w = w0 + p % COLS;
    if (h >= 0 && h < height && w >= 0 && w < width) {
      bn_relu_vector(vectors + (p - lane.first) * kVectors, scale, shift);
    }
  }
}

// relu(y * scale + shift) in place on the in-image pixels of a staged ROWS x COLS tile whose pixel (0, 0) is image
// pixel (h0, w0); the zeros the TMA filled in outside the image stay, they are the convolution's padding. The
// lanes loop with constant bounds; tiles inside the image skip the per-pixel checks. Ends with the block's sync.
template <int ROWS, int COLS, int WPG>
__device__ __forceinline__ void bn_relu_staged(unsigned char* tile, int h0, int w0, int height, int width,
                                               const float (&scale)[8], const float (&shift)[8],
                                               const VectorLane<WPG>& lane) {
  bn_relu_staged_pixels<ROWS, COLS, WPG>(tile, h0, w0, height, width, scale, shift, lane);
  fence_async_shared();
  lane.sync();
}

// Adds a tile's output sums, per thread in the mma accumulator layout, to the block's running sums in shared memory
// (sums[warp_in_group][72 channels][2]): reduced over the lanes, then lane l < 4 of each warp adds its channels to
// slots only it writes, so the order is fixed.
template <int WPG>
__device__ __forceinline__ void add_tile_sums(float* sums, float (&sum)[3][2], float (&square)[3][2]) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int group = warp / WPG;
  const int warp_in_group = warp % WPG;
#pragma unroll
  for (int nt = 0; nt < 3; ++nt) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
#pragma unroll
      for (int offset = 4; offset < 32; offset *= 2) {
        sum[nt][e] += __shfl_xor_sync(0xffffffff, sum[nt][e], offset);
        square[nt][e] += __shfl_xor_sync(0xffffffff, square[nt][e], offset);
      }
      if (lane < 4) {
        float* slot = sums + (warp_in_group * kBlockChannels + group * kGroup + nt * 8 + 2 * lane + e) * 2;
        slot[0] += sum[nt][e];
        slot[1] += square[nt][e];
      }
    }
  }
}

// Writes the block's output sums of add_tile_sums, adding the WPG warps of a group in a fixed order.
template <int WPG>
__device__ __forceinline__ void write_tile_sums(const float* sums, float* partial, int channels, int c0) {
  __syncthreads();
  if (threadIdx.x < 2 * kBlockChannels) {
    const int c = threadIdx.x % kBlockChannels;
    const int stat = threadIdx.x / kBlockChannels;
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < WPG; ++w) {
      total += sums[(w * kBlockChannels + c) * 2 + stat];
    }
    partial[(static_cast<int64_t>(blockIdx.x) * 2 + stat) * channels + c0 + c] = total;
  }
}

__device__ __forceinline__ float round_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// The weights of a 72-channel chunk as the stride-1 kernel holds them in shared memory: per group and tap t (staged
// offset (t / 3, t % 3)) B[k][n] as 24 rows n of 24 k, 48 bytes each (bank-conflict free for ldmatrix).
constexpr int kWeightRowBytes = kGroup * 2;
constexpr int kWeightTapBytes = kGroup * kWeightRowBytes;
constexpr int kWeightBytes = kGroupsPerBlock * kTaps * kWeightTapBytes;

// packed[c / 72] = the shared weight image of channel chunk c / 72, for the convolution (B[k = i][n = o] at tap) or,
// with TRANSPOSE, its input gradient (B[k = o][n = i] at the rotated tap), rounded to bf16 like autocast. One thread
// per 8 consecutive k.
template <bool TRANSPOSE, typename weight_t>
__global__ void __launch_bounds__(256) pack_weights_kernel(const weight_t* __restrict__ weight, WeightStrides strides,
                                                           uint4* __restrict__ packed, int vectors) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= vectors) {
    return;
  }
  // v enumerates (group, t, n, k / 8).
  const int k0 = (v % 3) * 8;
  const int n = (v / 3) % kGroup;
  const int t = (v / (3 * kGroup)) % kTaps;
  const int group_channel = (v / (3 * kGroup * kTaps)) * kGroup;
  const int tap = TRANSPOSE ? kTaps - 1 - t : t;
  __align__(16) __nv_bfloat16 values[8];
#pragma unroll
  for (int x = 0; x < 8; ++x) {
    const int k = k0 + x;
    const int index = TRANSPOSE ? strides.at(group_channel + k, n, tap) : strides.at(group_channel + n, k, tap);
    values[x] = __float2bfloat16_rn(to_float(weight[index]));
  }
  packed[v] = *reinterpret_cast<const uint4*>(values);
}

template <int TH, int STAGES>
struct S1Layout {
  using T = Tile<0, TH, 8>;
  static constexpr int kStageBytes = round_up(T::kSharedBytes);
  static constexpr int kOutBytes = round_up(TH * 8 * kPixelBytes);
  static constexpr int kOutOffset = STAGES * kStageBytes;
  static constexpr int kWeightOffset = kOutOffset + kOutBytes;
  static constexpr int kBarrierOffset = kWeightOffset + kWeightBytes;
  // The stages' barriers, then the fused input gradient's for y.
  static constexpr int kBytes = kBarrierOffset + (STAGES + 1) * 8;
};

// Stride-1 convolution of source into target through TMA maps with the weights of pack_weights_kernel in shared memory;
// with TRANSPOSE (packed weights transposed and rotated) the input gradient. Threads: 3 groups x WPG warps, each warp
// running all P = TH / (2 WPG) m16 tiles of its rows at once, tap by tap.
template <int TH, int WPG, int STAGES, bool TRANSPOSE, bool FUSED>
__global__ void __launch_bounds__(32 * kGroupsPerBlock * WPG, 2)
    grouped_conv_s1_kernel(const __grid_constant__ CUtensorMap source_map,
                           const __grid_constant__ CUtensorMap target_map, const __grid_constant__ CUtensorMap y_map,
                           const uint4* __restrict__ packed, int batch, int height, int width, int channels,
                           BnArgs bn) {
  constexpr int TW = 8;
  using T = Tile<0, TH, TW>;
  using L = S1Layout<TH, STAGES>;
  constexpr int P = TH / 2 / WPG;
  static_assert(TH % (2 * WPG) == 0, "the warps of a group share the m16 tiles evenly");
  extern __shared__ __align__(128) unsigned char shared[];

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int group = warp / WPG;
  const int warp_in_group = warp % WPG;
  const int c0 = blockIdx.y * kBlockChannels;
  const int q = lane >> 3;
  const bool leader = threadIdx.x == 0;

  const int tiles_w = (width + TW - 1) / TW;
  const int tiles_h = (height + TH - 1) / TH;
  const int tiles = batch * tiles_h * tiles_w;
  const uint32_t buffers = shared_address(shared);
  const uint32_t out_tile = buffers + L::kOutOffset;
  const uint32_t weights = buffers + L::kWeightOffset;
  const uint32_t barriers = buffers + L::kBarrierOffset;

  auto stage = [&](int slot, int tile, int extra_bytes) {
    mbarrier_expect(barriers + slot * 8, T::kBytes + extra_bytes);
    load_tile<0, TH, TW>(buffers + slot * L::kStageBytes, &source_map, c0, (tile / tiles_w) % tiles_h, tile % tiles_w,
                         tile / (tiles_w * tiles_h), barriers + slot * 8);
  };

  if (leader) {
#pragma unroll
    for (int slot = 0; slot < STAGES + 1; ++slot) {
      mbarrier_init(barriers + slot * 8, 1);
    }
    fence_mbarrier_init();
  }
  __syncthreads();
  int tile = blockIdx.x;
  if (leader && tile < tiles) {
    stage(0, tile, kWeightBytes);
    bulk_load(weights, packed + blockIdx.y * (kWeightBytes / 16), kWeightBytes, barriers);
#pragma unroll
    for (int ahead = 1; ahead < STAGES - 1; ++ahead) {
      if (tile + ahead * static_cast<int>(gridDim.x) < tiles) {
        stage(ahead, tile + ahead * gridDim.x, 0);
      }
    }
  }

  // The lane's ldmatrix rows of a tap's 9 weight matrices (nt, part): (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)
  // by the first x4 and half the second, then (0, 2), (1, 2), (2, 2) (registers 6..8; the third x4 repeats (2, 2)).
  uint32_t b_address[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int m = min(4 * j + q, 8);
    const int nt = m < 6 ? m / 2 : m - 6;
    const int part = m < 6 ? m % 2 : 2;
    b_address[j] = weights + group * kTaps * kWeightTapBytes + (nt * 8 + (lane & 7)) * kWeightRowBytes + part * 16;
  }
  // ldmatrix rows: lane l addresses pixel (l & 7) + 8 * ((l >> 3) & 1) of an m16 tile, channel chunk l >> 4 (k16) or 2.
  const uint32_t a_lane =
      ((((lane >> 3) & 1) * T::kCols + (lane & 7)) * kPixelBytes) + (group * kGroup + (lane >> 4) * 8) * 2;
  const int first_row = 2 * warp_in_group * P;
  const int n_lane = lane / 4;
  const int k_lane = 2 * (lane % 4);

  // FUSED: the threads of the staged tile's batch norm + ReLU, and the block's sums (see add_tile_sums): the output's
  // and its square's, or the input gradient's of shift and of scale.
  const VectorLane<WPG> vectors;
  __shared__ float sums[FUSED ? WPG * kBlockChannels * 2 : 1];
  if constexpr (FUSED) {
    for (int i = threadIdx.x; i < WPG * kBlockChannels * 2; i += blockDim.x) {
      sums[i] = 0.0f;
    }
    // The first tile's __syncthreads orders the zeros before the first add_tile_sums.
  }
  const uint32_t y_barrier = barriers + STAGES * 8;
  int y_parity = 0;

  int slot = 0;
  int parity = 0;
  for (; tile < tiles; tile += gridDim.x) {
    const int ahead = tile + (STAGES - 1) * static_cast<int>(gridDim.x);
    if (leader && ahead < tiles) {
      stage(slot == 0 ? STAGES - 1 : slot - 1, ahead, 0);
    }
    const int tile_w = tile % tiles_w;
    const int tile_h = (tile / tiles_w) % tiles_h;
    const int image = tile / (tiles_w * tiles_h);
    if (FUSED && TRANSPOSE && leader) {
      // The previous tile's store has read the output tile.
      tma_store_wait_read<0>();
      mbarrier_expect(y_barrier, TH * TW * kPixelBytes);
      tma_load(out_tile, &y_map, c0, tile_w * TW, tile_h * TH, image, y_barrier);
    }
    mbarrier_wait(barriers + slot * 8, parity);
    if constexpr (FUSED && !TRANSPOSE) {
      float scale[8];
      float shift[8];
      load_channels(bn.scale, c0 + vectors.vector * 8, scale);
      load_channels(bn.shift, c0 + vectors.vector * 8, shift);
      bn_relu_staged<T::kRows, T::kCols, WPG>(shared + slot * L::kStageBytes, tile_h * TH - 1, tile_w * TW - 1, height,
                                              width, scale, shift, vectors);
    }

    // Taps in cuDNN's order (the kernel's taps ascending; the input gradient's staged taps descending), each as
    // channels 0..15 then 16..23.
    const uint32_t row_address = buffers + slot * L::kStageBytes + first_row * T::kCols * kPixelBytes + a_lane;
    float acc[P][3][4] = {};
#pragma unroll
    for (int step = 0; step < kTaps; ++step) {
      const int t = TRANSPOSE ? kTaps - 1 - step : step;
      uint32_t bq[12];
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        ldmatrix_x4(*reinterpret_cast<uint32_t (*)[4]>(bq + 4 * j), b_address[j] + t * kWeightTapBytes);
      }
#pragma unroll
      for (int p = 0; p < P; ++p) {
        const uint32_t address = row_address + ((2 * p + t / 3) * T::kCols + t % 3) * kPixelBytes;
        uint32_t a[4];
        uint32_t a8[2];
        ldmatrix_x4(a, address);
        ldmatrix_x2(a8, address + (2 - (lane >> 4)) * 16);
#pragma unroll
        for (int nt = 0; nt < 3; ++nt) {
          mma(acc[p][nt], a, bq[2 * nt], bq[2 * nt + 1]);
          mma_k8(acc[p][nt], a8, bq[6 + nt]);
        }
      }
    }
    // The previous tile's store has read the output tile; every warp is done with this stage buffer.
    if (leader) {
      tma_store_wait_read<0>();
    }
    __syncthreads();
    float first_sum[3][2] = {};
    float second_sum[3][2] = {};
    // FUSED input gradient: the 6 channels of the lane's accumulators, c = group * 24 + nt * 8 + k_lane + e.
    float2 scale[3];
    float2 shift[3];
    float2 center[3];
    if constexpr (FUSED && TRANSPOSE) {
#pragma unroll
      for (int nt = 0; nt < 3; ++nt) {
        const int c = c0 + group * kGroup + nt * 8 + k_lane;
        scale[nt] = __ldg(reinterpret_cast<const float2*>(bn.scale + c));
        shift[nt] = __ldg(reinterpret_cast<const float2*>(bn.shift + c));
        center[nt] = __ldg(reinterpret_cast<const float2*>(bn.center + c));
      }
      mbarrier_wait(y_barrier, y_parity);
      y_parity ^= 1;
    }
#pragma unroll
    for (int p = 0; p < P; ++p) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const uint32_t out =
            out_tile + ((first_row + 2 * p + half) * TW + n_lane) * kPixelBytes + (group * kGroup + k_lane) * 2;
        const bool inside = tile_h * TH + first_row + 2 * p + half < height && tile_w * TW + n_lane < width;
        if constexpr (FUSED && TRANSPOSE) {
          // y's pair in the output tile becomes the gradient of y: g (the gradient of relu(y * scale + shift),
          // rounded to bf16 like the unfused input gradient) through the ReLU and the scale.
#pragma unroll
          for (int nt = 0; nt < 3; ++nt) {
            uint32_t y_bits;
            asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(y_bits) : "r"(out + nt * 16));
            const float2 y = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&y_bits));
            float g0 = round_bf16(acc[p][nt][2 * half]);
            float g1 = round_bf16(acc[p][nt][2 * half + 1]);
            g0 = fmaf(y.x, scale[nt].x, shift[nt].x) > 0.0f && inside ? g0 : 0.0f;
            g1 = fmaf(y.y, scale[nt].y, shift[nt].y) > 0.0f && inside ? g1 : 0.0f;
            first_sum[nt][0] += g0;
            first_sum[nt][1] += g1;
            second_sum[nt][0] = fmaf(g0, y.x - center[nt].x, second_sum[nt][0]);
            second_sum[nt][1] = fmaf(g1, y.y - center[nt].y, second_sum[nt][1]);
            store_shared(out + nt * 16, g0 * scale[nt].x, g1 * scale[nt].y);
          }
        } else {
#pragma unroll
          for (int nt = 0; nt < 3; ++nt) {
            store_shared(out + nt * 16, acc[p][nt][2 * half], acc[p][nt][2 * half + 1]);
          }
        }
        if constexpr (FUSED && !TRANSPOSE) {
          if (inside) {
#pragma unroll
            for (int nt = 0; nt < 3; ++nt) {
#pragma unroll
              for (int e = 0; e < 2; ++e) {
                const float v = round_bf16(acc[p][nt][2 * half + e]);
                first_sum[nt][e] += v;
                second_sum[nt][e] = fmaf(v, v, second_sum[nt][e]);
              }
            }
          }
        }
      }
    }
    if constexpr (FUSED) {
      add_tile_sums<WPG>(sums, first_sum, second_sum);
    }
    fence_async_shared();
    __syncthreads();
    if (leader) {
      tma_store(&target_map, c0, tile_w * TW, tile_h * TH, image, out_tile);
      tma_store_commit();
    }
    if (++slot == STAGES) {
      slot = 0;
      parity ^= 1;
    }
  }
  if (leader) {
    tma_store_wait_read<0>();
  }
  if constexpr (FUSED) {
    write_tile_sums<WPG>(sums, bn.partial, channels, c0);
  }
}

// The output tile of MODE 1, double buffered so a tile's TMA store overlaps the next tile, or with MODE 2 its 2TH x 2TW
// input-gradient pixels, which the threads copy out.
template <int MODE, int TH, int STAGES>
struct S2Layout {
  using T = Tile<MODE, TH, 8>;
  static constexpr int kStageBytes = round_up(T::kSharedBytes);
  // MODE 1: a tile's TMA store overlaps the next tile; MODE 2 FUSED: the next tile's y loads during this tile.
  static constexpr int kOutBuffers = 2;
  static constexpr int kOutBytes = round_up((MODE == 2 ? 4 : 1) * TH * 8 * kPixelBytes);
  static constexpr int kOutOffset = STAGES * kStageBytes;
  static constexpr int kBarrierOffset = kOutOffset + kOutBuffers * kOutBytes;
  // The stages' barriers, then the fused input gradient's for y, one per output buffer.
  static constexpr int kBytes = kBarrierOffset + (STAGES + 2) * 8;
};

// Threads: 3 groups x WPG warps; blocks: persistent over the tiles x 72-channel chunks. MODE 1 is the stride-2
// convolution, MODE 2 its input gradient (the weights transposed). The tiles cover target pixels (MODE 1, stored with
// target_map) or 2 x 2 target cells (MODE 2, copied out to target).
template <int MODE, int TH, int WPG, int STAGES, typename weight_t, bool FUSED>
__global__ void __launch_bounds__(32 * kGroupsPerBlock * WPG, 2)
    grouped_conv_s2_kernel(const __grid_constant__ CUtensorMap source_map,
                           const __grid_constant__ CUtensorMap target_map, const __grid_constant__ CUtensorMap y_map,
                           const weight_t* __restrict__ weight, WeightStrides strides,
                           __nv_bfloat16* __restrict__ target, int target_height, int target_width, int channels,
                           int batch, int cells_height, int cells_width, BnArgs bn) {
  constexpr int TW = 8;
  constexpr bool TRANSPOSE = MODE == 2;
  using T = Tile<MODE, TH, TW>;
  using L = S2Layout<MODE, TH, STAGES>;
  constexpr int kTileM = T::kPixels / 16;
  constexpr int kWarpM = kTileM / WPG;
  static_assert(kTileM % WPG == 0, "the warps of a group share the m16 tiles evenly");
  extern __shared__ __align__(128) unsigned char shared[];

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int group = warp / WPG;
  const int warp_in_group = warp % WPG;
  const int c0 = blockIdx.y * kBlockChannels;
  const int group_channel = c0 + group * kGroup;
  const bool leader = threadIdx.x == 0;

  const int tiles_w = (cells_width + TW - 1) / TW;
  const int tiles_h = (cells_height + TH - 1) / TH;
  const int tiles = batch * tiles_h * tiles_w;
  const uint32_t buffers = shared_address(shared);
  const uint32_t barriers = buffers + L::kBarrierOffset;

  auto stage = [&](int slot, int tile) {
    mbarrier_expect(barriers + slot * 8, T::kBytes);
    load_tile<MODE, TH, TW>(buffers + slot * L::kStageBytes, &source_map, c0, (tile / tiles_w) % tiles_h,
                            tile % tiles_w, tile / (tiles_w * tiles_h), barriers + slot * 8);
  };

  if (leader) {
#pragma unroll
    for (int slot = 0; slot < STAGES + 2; ++slot) {
      mbarrier_init(barriers + slot * 8, 1);
    }
    fence_mbarrier_init();
  }
  __syncthreads();
  int tile = blockIdx.x;
  if (leader) {
#pragma unroll
    for (int ahead = 0; ahead < STAGES - 1; ++ahead) {
      if (tile + ahead * static_cast<int>(gridDim.x) < tiles) {
        stage(ahead, tile + ahead * gridDim.x);
      }
    }
  }

  const int n_lane = lane / 4;
  const int k_lane = 2 * (lane % 4);
  // B[k = (tap, j)][n] for the 3 n8 tiles: k 0..15 of a tap as two k16 halves, k 16..23 as one k8.
  uint32_t b[kTaps][9];
#pragma unroll
  for (int t = 0; t < kTaps; ++t) {
    const int tap = weight_tap<MODE>(t);
#pragma unroll
    for (int nt = 0; nt < 3; ++nt) {
      const int n = nt * 8 + n_lane;
#pragma unroll
      for (int part = 0; part < 3; ++part) {
        const int j = k_lane + part * 8;
        // weight[o][i][tap] with o the output and i the input channel of the convolution.
        const int first = TRANSPOSE ? strides.at(group_channel + j, n, tap) : strides.at(group_channel + n, j, tap);
        const int second = first + (TRANSPOSE ? strides.o : strides.i);
        b[t][nt * 3 + part] = weight_pair(weight, first, second);
      }
    }
  }

  // ldmatrix rows: lane l addresses pixel (l & 7) + 8 * ((l >> 3) & 1) of an m16 tile, channel chunk l >> 4 (k16) or 2.
  const int lane_row = (lane >> 3) & 1;
  const int lane_col = lane & 7;
  const uint32_t lane_bytes = (group * kGroup + (lane >> 4) * 8) * 2;

  // FUSED: as in grouped_conv_s1_kernel.
  const VectorLane<WPG> vectors;
  __shared__ float sums[FUSED ? WPG * kBlockChannels * 2 : 1];
  // MODE 2 FUSED: y's 2TH x 2TW pixels of a tile load into its output buffer one tile ahead, on that buffer's barrier.
  const uint32_t y_barriers = barriers + STAGES * 8;
  int y_parities = 0;
  auto load_y = [&](int next, int buffer) {
    mbarrier_expect(y_barriers + buffer * 8, 4 * TH * TW * kPixelBytes);
    tma_load(buffers + L::kOutOffset + buffer * L::kOutBytes, &y_map, c0, 2 * (next % tiles_w) * TW,
             2 * ((next / tiles_w) % tiles_h) * TH, next / (tiles_w * tiles_h), y_barriers + buffer * 8);
  };
  if (FUSED && MODE == 2 && leader && tile < tiles) {
    load_y(tile, 0);
  }
  if constexpr (FUSED) {
    for (int i = threadIdx.x; i < WPG * kBlockChannels * 2; i += blockDim.x) {
      sums[i] = 0.0f;
    }
    // The first tile's __syncthreads orders the zeros before the first add_tile_sums.
  }

  int slot = 0;
  int parity = 0;
  int out_buffer = 0;
  for (; tile < tiles; tile += gridDim.x) {
    const int ahead = tile + (STAGES - 1) * static_cast<int>(gridDim.x);
    if (leader) {
      if (ahead < tiles) {
        stage(slot == 0 ? STAGES - 1 : slot - 1, ahead);
      }
      // The store of two tiles ago has read this output buffer.
      if (MODE == 1) {
        tma_store_wait_read<1>();
      }
    }
    __syncthreads();
    const int tile_w = tile % tiles_w;
    const int tile_h = (tile / tiles_w) % tiles_h;
    const int image = tile / (tiles_w * tiles_h);
    if (FUSED && MODE == 2 && leader && tile + static_cast<int>(gridDim.x) < tiles) {
      // The other buffer: the previous tile's copy-out has read it.
      load_y(tile + gridDim.x, out_buffer ^ 1);
    }
    mbarrier_wait(barriers + slot * 8, parity);
    if constexpr (FUSED && MODE == 1) {
      float scale[8];
      float shift[8];
      load_channels(bn.scale, c0 + vectors.vector * 8, scale);
      load_channels(bn.shift, c0 + vectors.vector * 8, shift);
      bn_relu_staged<T::kRows, T::kCols, WPG>(shared + slot * L::kStageBytes, 2 * tile_h * TH - 1, 2 * tile_w * TW - 1,
                                              bn.height, bn.width, scale, shift, vectors);
    }

    const uint32_t tile_address = buffers + slot * L::kStageBytes;
    const uint32_t out_tile = buffers + L::kOutOffset + out_buffer * L::kOutBytes;
    constexpr int kPhases = MODE == 2 ? 4 : 1;
    float first_sum[3][2] = {};
    float second_sum[3][2] = {};
    // FUSED input gradient: the 6 channels of the lane's accumulators, as in grouped_conv_s1_kernel.
    float2 scale[3];
    float2 shift[3];
    float2 center[3];
    if constexpr (FUSED && MODE == 2) {
#pragma unroll
      for (int nt = 0; nt < 3; ++nt) {
        const int c = group_channel + nt * 8 + k_lane;
        scale[nt] = __ldg(reinterpret_cast<const float2*>(bn.scale + c));
        shift[nt] = __ldg(reinterpret_cast<const float2*>(bn.shift + c));
        center[nt] = __ldg(reinterpret_cast<const float2*>(bn.center + c));
      }
    }
#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
      const int t_begin = MODE == 2 ? phase_start(phase) : 0;
      const int t_end = MODE == 2 ? phase_start(phase + 1) : kTaps;
#pragma unroll
      for (int pair = 0; pair < (kWarpM + 1) / 2; ++pair) {
        const int m_first = warp_in_group * kWarpM + pair * 2;
        const int count = min(2, kWarpM - pair * 2);
        float acc[2][3][4] = {};
#pragma unroll
        for (int t = t_begin; t < t_end; ++t) {
          const uint32_t offset = tap_offset<MODE, TH, TW>(t) * kPixelBytes;
#pragma unroll
          for (int m = 0; m < count; ++m) {
            const int row = 2 * (m_first + m) + lane_row;
            const uint32_t address = tile_address + T::base(row, lane_col) * kPixelBytes + offset + lane_bytes;
            uint32_t a[4];
            uint32_t a8[2];
            ldmatrix_x4(a, address);
            ldmatrix_x2(a8, address + (2 - (lane >> 4)) * 16);
#pragma unroll
            for (int nt = 0; nt < 3; ++nt) {
              mma(acc[m][nt], a, b[t][nt * 3], b[t][nt * 3 + 1]);
              mma_k8(acc[m][nt], a8, b[t][nt * 3 + 2]);
            }
          }
        }
#pragma unroll
        for (int m = 0; m < count; ++m) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int row = 2 * (m_first + m) + half;
            const int pixel = MODE == 2 ? (2 * row + phase / 2) * 2 * TW + 2 * n_lane + phase % 2 : row * TW + n_lane;
            const uint32_t out = out_tile + pixel * kPixelBytes + (group * kGroup + k_lane) * 2;
            if constexpr (FUSED && MODE == 2) {
              if (phase == 0 && pair == 0 && m == 0 && half == 0) {
                mbarrier_wait(y_barriers + out_buffer * 8, (y_parities >> out_buffer) & 1);
                y_parities ^= 1 << out_buffer;
              }
              const bool inside = 2 * tile_h * TH + pixel / (2 * TW) < target_height &&
                                  2 * tile_w * TW + pixel % (2 * TW) < target_width;
#pragma unroll
              for (int nt = 0; nt < 3; ++nt) {
                uint32_t y_bits;
                asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(y_bits) : "r"(out + nt * 16));
                const float2 y = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&y_bits));
                float g0 = round_bf16(acc[m][nt][2 * half]);
                float g1 = round_bf16(acc[m][nt][2 * half + 1]);
                g0 = fmaf(y.x, scale[nt].x, shift[nt].x) > 0.0f && inside ? g0 : 0.0f;
                g1 = fmaf(y.y, scale[nt].y, shift[nt].y) > 0.0f && inside ? g1 : 0.0f;
                first_sum[nt][0] += g0;
                first_sum[nt][1] += g1;
                second_sum[nt][0] = fmaf(g0, y.x - center[nt].x, second_sum[nt][0]);
                second_sum[nt][1] = fmaf(g1, y.y - center[nt].y, second_sum[nt][1]);
                store_shared(out + nt * 16, g0 * scale[nt].x, g1 * scale[nt].y);
              }
            } else {
#pragma unroll
              for (int nt = 0; nt < 3; ++nt) {
                store_shared(out + nt * 16, acc[m][nt][2 * half], acc[m][nt][2 * half + 1]);
              }
            }
            if constexpr (FUSED && MODE == 1) {
              if (tile_h * TH + row < target_height && tile_w * TW + n_lane < target_width) {
#pragma unroll
                for (int nt = 0; nt < 3; ++nt) {
#pragma unroll
                  for (int e = 0; e < 2; ++e) {
                    const float v = round_bf16(acc[m][nt][2 * half + e]);
                    first_sum[nt][e] += v;
                    second_sum[nt][e] = fmaf(v, v, second_sum[nt][e]);
                  }
                }
              }
            }
          }
        }
      }
    }
    if constexpr (FUSED) {
      add_tile_sums<WPG>(sums, first_sum, second_sum);
    }
    fence_async_shared();
    __syncthreads();
    if (MODE == 2) {
      const int n = image;
      const unsigned char* out_bytes = shared + L::kOutOffset + out_buffer * L::kOutBytes;
      for (int i = threadIdx.x; i < 2 * TH * 2 * TW * (kPixelBytes / 16); i += blockDim.x) {
        const int chunk = i % (kPixelBytes / 16);
        const int pixel = i / (kPixelBytes / 16);
        const int h = 2 * tile_h * TH + pixel / (2 * TW);
        const int w = 2 * tile_w * TW + pixel % (2 * TW);
        if (h < target_height && w < target_width) {
          *reinterpret_cast<uint4*>(
              target + ((static_cast<int64_t>(n) * target_height + h) * target_width + w) * channels + c0 + chunk * 8) =
              *reinterpret_cast<const uint4*>(out_bytes + i * 16);
        }
      }
      // The next tile's TMA load of y overwrites what the copy-out read.
      fence_async_shared();
    } else if (leader) {
      const int factor = MODE == 2 ? 2 : 1;
      tma_store(&target_map, c0, factor * tile_w * TW, factor * tile_h * TH, image, out_tile);
      tma_store_commit();
    }
    out_buffer = (out_buffer + 1) % L::kOutBuffers;
    if (++slot == STAGES) {
      slot = 0;
      parity ^= 1;
    }
  }
  if (MODE == 1 && leader) {
    tma_store_wait_read<0>();
  }
  if constexpr (FUSED) {
    write_tile_sums<WPG>(sums, bn.partial, channels, c0);
  }
}

template <int MODE, int TH, int STAGES>
struct GradLayout {
  using T = Tile<MODE, TH, 8>;
  static constexpr int kGradOffset = round_up(T::kSharedBytes);
  static constexpr int kStageBytes = kGradOffset + round_up(TH * 8 * kPixelBytes);
  static constexpr int kBarrierOffset = STAGES * kStageBytes;
  static constexpr int kBytes = kBarrierOffset + STAGES * 8;
};

// Threads: 3 groups x WPG warps; blocks: (pixel chunks, 72-channel chunks), each over a contiguous run of tiles, in
// clusters of kGradCluster along the pixel chunks. Writes partial[cluster][c][tap * 24 + i] = sum over the cluster's
// tiles' output pixels p of grad[p][c] * x[p + tap][i] (fp32). Out-of-image pixels stage as zeros and add nothing.
// With FUSED, x is the batch norm's input and the staged tiles become relu(x * scale + shift) (see BnArgs).
template <int MODE, int TH, int WPG, int STAGES, bool FUSED>
__global__ void __launch_bounds__(32 * kGroupsPerBlock * WPG, 2)
    grouped_conv_weight_grad_kernel(const __grid_constant__ CUtensorMap x_map,
                                    const __grid_constant__ CUtensorMap grad_map, float* __restrict__ partial,
                                    int batch, int out_height, int out_width, int channels, BnArgs bn) {
  constexpr int TW = 8;
  using T = Tile<MODE, TH, TW>;
  using L = GradLayout<MODE, TH, STAGES>;
  constexpr int kGradBytes = T::kPixels * kPixelBytes;
  constexpr int kWarpM = (kGradTiles + WPG - 1) / WPG;
  extern __shared__ __align__(128) unsigned char shared[];

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int group = warp / WPG;
  const int warp_in_group = warp % WPG;
  const int c0 = blockIdx.y * kBlockChannels;
  const int q = lane >> 3;
  const bool leader = threadIdx.x == 0;

  // A = x^T: rows (tap, i) in 8-channel chunks, two per m16 tile; the last tile's second chunk repeats the first.
  uint32_t a_offset[kWarpM];
#pragma unroll
  for (int j = 0; j < kWarpM; ++j) {
    int chunk = 2 * (warp_in_group * kWarpM + j) + (q & 1);
    chunk = min(chunk, kGradChunks - 1);
    a_offset[j] = tap_offset<MODE, TH, TW>(chunk / 3) * kPixelBytes + (group * kGroup + (chunk % 3) * 8) * 2;
  }
  // ldmatrix.trans rows: the A tiles' lane reads pixel (l & 7) + 8 * (l >> 4), the B tiles' pixel (l & 7) + 8 * (q &
  // 1).
  const uint32_t a_lane = T::base(q >> 1, lane & 7) * kPixelBytes;
  const uint32_t b_lane = ((q & 1) * TW + (lane & 7)) * kPixelBytes + (group * kGroup + (q >> 1) * 8) * 2;
  const uint32_t b8_lane = ((q & 1) * TW + (lane & 7)) * kPixelBytes + (group * kGroup + 16) * 2;

  const int tiles_w = (out_width + TW - 1) / TW;
  const int tiles_h = (out_height + TH - 1) / TH;
  const int tiles = batch * tiles_h * tiles_w;
  const int per_block = (tiles + gridDim.x - 1) / gridDim.x;
  const int begin = blockIdx.x * per_block;
  const int end = min(begin + per_block, tiles);
  const uint32_t buffers = shared_address(shared);
  const uint32_t barriers = buffers + L::kBarrierOffset;

  auto stage = [&](int slot, int tile) {
    const int tile_w = tile % tiles_w;
    const int tile_h = (tile / tiles_w) % tiles_h;
    const int n = tile / (tiles_w * tiles_h);
    const uint32_t destination = buffers + slot * L::kStageBytes;
    mbarrier_expect(barriers + slot * 8, T::kBytes + kGradBytes);
    load_tile<MODE, TH, TW>(destination, &x_map, c0, tile_h, tile_w, n, barriers + slot * 8);
    tma_load(destination + L::kGradOffset, &grad_map, c0, tile_w * TW, tile_h * TH, n, barriers + slot * 8);
  };

  if (leader) {
#pragma unroll
    for (int slot = 0; slot < STAGES; ++slot) {
      mbarrier_init(barriers + slot * 8, 1);
    }
    fence_mbarrier_init();
  }
  __syncthreads();
  if (leader) {
#pragma unroll
    for (int ahead = 0; ahead < STAGES - 1; ++ahead) {
      if (begin + ahead < end) {
        stage(ahead, begin + ahead);
      }
    }
  }

  const VectorLane<WPG> vectors;

  float acc[kWarpM][3][4] = {};
  int slot = 0;
  int parity = 0;
  for (int tile = begin; tile < end; ++tile) {
    if (leader && tile + STAGES - 1 < end) {
      stage(slot == 0 ? STAGES - 1 : slot - 1, tile + STAGES - 1);
    }
    mbarrier_wait(barriers + slot * 8, parity);
    if constexpr (FUSED) {
      const int tile_w = tile % tiles_w;
      const int tile_h = (tile / tiles_w) % tiles_h;
      float scale[8];
      float shift[8];
      load_channels(bn.scale, c0 + vectors.vector * 8, scale);
      load_channels(bn.shift, c0 + vectors.vector * 8, shift);
      bn_relu_staged<T::kRows, T::kCols, WPG>(shared + slot * L::kStageBytes, T::kScale * tile_h * TH - T::kShift,
                                              T::kScale * tile_w * TW - T::kShift, bn.height, bn.width, scale, shift,
                                              vectors);
    }
    const uint32_t x_tile = buffers + slot * L::kStageBytes;
    const uint32_t grad_tile = x_tile + L::kGradOffset;
#pragma unroll 2
    for (int step = 0; step < T::kPixels / 16; ++step) {
      uint32_t b[4];
      uint32_t b8[2];
      ldmatrix_x4_trans(b, grad_tile + step * 16 * kPixelBytes + b_lane);
      ldmatrix_x2_trans(b8, grad_tile + step * 16 * kPixelBytes + b8_lane);
      const uint32_t x_step = x_tile + a_lane + T::base(2 * step, 0) * kPixelBytes;
#pragma unroll
      for (int j = 0; j < kWarpM; ++j) {
        if (warp_in_group * kWarpM + j < kGradTiles) {
          uint32_t a[4];
          ldmatrix_x4_trans(a, x_step + a_offset[j]);
          mma(acc[j][0], a, b[0], b[1]);
          mma(acc[j][1], a, b[2], b[3]);
          mma(acc[j][2], a, b8[0], b8[1]);
        }
      }
    }
    __syncthreads();
    if (++slot == STAGES) {
      slot = 0;
      parity ^= 1;
    }
  }

  // The block's sums into its shared memory as [72 channels][kGradRows]; then each block of the cluster adds its
  // 1 / kGradCluster share of them over the cluster's blocks, in rank order, through distributed shared memory, into
  // partial[cluster][c][tap * 24 + i].
  float* block_sums = reinterpret_cast<float*>(shared);
  const int n_lane = 2 * (lane % 4);
  const int row_lane = lane / 4;
#pragma unroll
  for (int j = 0; j < kWarpM; ++j) {
    const int m_tile = warp_in_group * kWarpM + j;
    if (m_tile >= kGradTiles) {
      continue;
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int chunk = 2 * m_tile + half;
      if (chunk >= kGradChunks) {
        continue;
      }
      const int row = chunk * 8 + row_lane;
#pragma unroll
      for (int nt = 0; nt < 3; ++nt) {
        const int o = group * kGroup + nt * 8 + n_lane;
        block_sums[o * kGradRows + row] = acc[j][nt][2 * half];
        block_sums[(o + 1) * kGradRows + row] = acc[j][nt][2 * half + 1];
      }
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  namespace cg = cooperative_groups;
  const cg::cluster_group cluster = cg::this_cluster();
  cluster.sync();
  constexpr int kShare = kBlockChannels * kGradRows / kGradCluster;
  static_assert(kBlockChannels * kGradRows % (kGradCluster * 4) == 0, "the cluster's shares split evenly");
  const int rank = static_cast<int>(cluster.block_rank());
  float* out = partial + static_cast<int64_t>(blockIdx.x / kGradCluster) * channels * kGradRows +
               static_cast<int64_t>(c0) * kGradRows + rank * kShare;
  for (int i = threadIdx.x * 4; i < kShare; i += blockDim.x * 4) {
    float4 total = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int r = 0; r < kGradCluster; ++r) {
      const float4 v = *reinterpret_cast<const float4*>(cluster.map_shared_rank(block_sums + rank * kShare + i, r));
      total.x += v.x;
      total.y += v.y;
      total.z += v.z;
      total.w += v.w;
    }
    *reinterpret_cast<float4*>(out + i) = total;
  }
  // No block leaves while the others read its shared memory.
  cluster.sync();
#else
  __trap();
#endif
}

// grad_weight[o][i][tap] = sum over blocks b of partial[b][o][tap * 24 + i], in the order of b. Threads: 32 rows x 8
// strided block subsets, combined in a fixed tree.
template <typename weight_t>
__global__ void __launch_bounds__(256)
    grouped_conv_weight_grad_reduce_kernel(const float* __restrict__ partial, weight_t* __restrict__ grad_weight,
                                           WeightStrides strides, int blocks, int rows) {
  __shared__ float sums[8][33];
  const int row = blockIdx.x * 32 + threadIdx.x;
  float sum = 0.0f;
  if (row < rows) {
    for (int b = threadIdx.y; b < blocks; b += 8) {
      sum += partial[static_cast<int64_t>(b) * rows + row];
    }
  }
  sums[threadIdx.y][threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.y == 0 && row < rows) {
    const float total =
        ((sums[0][threadIdx.x] + sums[1][threadIdx.x]) + (sums[2][threadIdx.x] + sums[3][threadIdx.x])) +
        ((sums[4][threadIdx.x] + sums[5][threadIdx.x]) + (sums[6][threadIdx.x] + sums[7][threadIdx.x]));
    const int o = row / kGradRows;
    const int k = row % kGradRows;
    store(grad_weight + strides.at(o, k % kGroup, k / kGroup), total);
  }
}

// out[r] = sum over blocks b of partial[b][r], in the order of b; threads as in grouped_conv_weight_grad_reduce_kernel.
__global__ void __launch_bounds__(256)
    column_sum_kernel(const float* __restrict__ partial, float* __restrict__ out, int blocks, int rows) {
  __shared__ float sums[8][33];
  const int row = blockIdx.x * 32 + threadIdx.x;
  float sum = 0.0f;
  if (row < rows) {
    for (int b = threadIdx.y; b < blocks; b += 8) {
      sum += partial[static_cast<int64_t>(b) * rows + row];
    }
  }
  sums[threadIdx.y][threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.y == 0 && row < rows) {
    out[row] = ((sums[0][threadIdx.x] + sums[1][threadIdx.x]) + (sums[2][threadIdx.x] + sums[3][threadIdx.x])) +
               ((sums[4][threadIdx.x] + sums[5][threadIdx.x]) + (sums[6][threadIdx.x] + sums[7][threadIdx.x]));
  }
}

// Blocks that fit on the GPU at once, cached per (kernel, threads, bytes); also raises the shared memory limit.
template <typename Kernel>
int resident_blocks(Kernel kernel, int threads, int bytes) {
  static std::mutex mutex;
  static std::map<std::tuple<const void*, int, int>, int> cache;
  const std::lock_guard<std::mutex> lock(mutex);
  const auto key = std::make_tuple(reinterpret_cast<const void*>(kernel), threads, bytes);
  auto found = cache.find(key);
  if (found != cache.end()) {
    return found->second;
  }
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(static_cast<size_t>(bytes) <= properties->sharedMemPerBlockOptin, "regnet_grouped_conv needs ", bytes,
              " bytes of shared memory per block; ", properties->name, " allows ", properties->sharedMemPerBlockOptin);
  if (bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  }
  int per_sm = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, bytes));
  const int blocks = std::max(1, per_sm) * properties->multiProcessorCount;
  cache.emplace(key, blocks);
  return blocks;
}

WeightStrides strides_of(const at::Tensor& weight) {
  return {static_cast<int>(weight.stride(0)), static_cast<int>(weight.stride(1)), static_cast<int>(weight.stride(2)),
          static_cast<int>(weight.stride(3))};
}

void check_inputs(const at::Tensor& x, const at::Tensor& weight) {
  const auto* properties = at::cuda::getDeviceProperties(x.device().index());
  TORCH_CHECK(properties->major >= 9, "regnet_grouped_conv needs TMA and thread block clusters (sm_90 or newer); ",
              properties->name, " is sm_", properties->major, properties->minor);
  TORCH_CHECK(x.is_cuda() && x.dim() == 4 && x.scalar_type() == at::kBFloat16,
              "regnet_grouped_conv: x must be a 4D "
              "bf16 CUDA tensor");
  TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast), "regnet_grouped_conv: x must be channels-last");
  const int64_t channels = x.size(1);
  TORCH_CHECK(channels % kBlockChannels == 0, "regnet_grouped_conv: channels must be a multiple of 72, got ", channels);
  TORCH_CHECK(weight.dim() == 4 && weight.size(0) == channels && weight.size(1) == kGroup && weight.size(2) == 3 &&
                  weight.size(3) == 3,
              "regnet_grouped_conv: weight must be a [C, 24, 3, 3] tensor");
  TORCH_CHECK(weight.scalar_type() == at::kFloat || weight.scalar_type() == at::kBFloat16,
              "regnet_grouped_conv: weight must be fp32 or bf16");
  TORCH_CHECK(weight.is_cuda() && weight.device() == x.device(), "regnet_grouped_conv: weight must be on x's device");
  TORCH_CHECK(x.numel() < (int64_t{1} << 31), "regnet_grouped_conv: x is too large");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "regnet_grouped_conv: x must be 16-byte aligned");
}

int64_t output_size(int64_t size, int64_t stride) { return (size - 1) / stride + 1; }

// The tile height among the candidates that pads `rows` the least, the largest on ties (the least halo).
int tile_height(int64_t rows, std::initializer_list<int> candidates) {
  int best = 0;
  int64_t best_rows = 0;
  for (const int height : candidates) {
    const int64_t padded = (rows + height - 1) / height * height;
    if (best == 0 || padded < best_rows || (padded == best_rows && height > best)) {
      best = height;
      best_rows = padded;
    }
  }
  return best;
}

// A 4D tensor map of the channels-last bf16 tensor t, (C, W, H, N) innermost first, with boxes of 72 channels x
// box_w x box_h pixels of one image; reads outside the tensor return zeros.
CUtensorMap tensor_map(const at::Tensor& t, int box_w, int box_h) {
  const cuuint64_t dims[4] = {static_cast<cuuint64_t>(t.size(1)), static_cast<cuuint64_t>(t.size(3)),
                              static_cast<cuuint64_t>(t.size(2)), static_cast<cuuint64_t>(t.size(0))};
  const cuuint64_t strides[3] = {static_cast<cuuint64_t>(t.stride(3) * 2), static_cast<cuuint64_t>(t.stride(2) * 2),
                                 static_cast<cuuint64_t>(t.stride(0) * 2)};
  const cuuint32_t box[4] = {static_cast<cuuint32_t>(kBlockChannels), static_cast<cuuint32_t>(box_w),
                             static_cast<cuuint32_t>(box_h), 1};
  return encode_tiled(t, 4, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                      "regnet_grouped_conv");
}

// The staged source tile of MODE as tensor map boxes of t.
template <int MODE, int TH>
CUtensorMap source_map(const at::Tensor& t) {
  using T = Tile<MODE, TH, 8>;
  return tensor_map(t, T::kCols, T::kRows);
}

// The weights as the stride-1 kernel stages them, [C / 72][kWeightBytes] (see pack_weights_kernel).
template <bool TRANSPOSE>
at::Tensor pack_weights(const at::Tensor& weight) {
  const int64_t channels = weight.size(0);
  auto packed = at::empty({channels / kBlockChannels * kWeightBytes / 2}, weight.options().dtype(at::kBFloat16));
  const int vectors = static_cast<int>(channels * kGroup * kTaps / 8);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto out = reinterpret_cast<uint4*>(packed.data_ptr<at::BFloat16>());
  if (weight.scalar_type() == at::kFloat) {
    pack_weights_kernel<TRANSPOSE>
        <<<(vectors + 255) / 256, 256, 0, stream>>>(weight.data_ptr<float>(), strides_of(weight), out, vectors);
  } else {
    pack_weights_kernel<TRANSPOSE><<<(vectors + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()), strides_of(weight), out, vectors);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return packed;
}

// The per-block sums of a fused kernel launched with `blocks` x chunks blocks, [blocks][2][C], added up into [2, C].
at::Tensor block_sums(const at::Tensor& partial) {
  auto sums = at::empty({2, partial.size(2)}, partial.options());
  const int rows = static_cast<int>(2 * partial.size(2));
  column_sum_kernel<<<(rows + 31) / 32, dim3(32, 8), 0, at::cuda::getCurrentCUDAStream()>>>(
      partial.data_ptr<float>(), sums.data_ptr<float>(), static_cast<int>(partial.size(0)), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return sums;
}

// FUSED: the batch norm of the source (see BnArgs); the sums go to *sums.
struct Fusion {
  const at::Tensor* scale = nullptr;
  const at::Tensor* shift = nullptr;
  const at::Tensor* center = nullptr;
  const at::Tensor* y = nullptr;
  at::Tensor* sums = nullptr;
};

BnArgs bn_args(const Fusion& fusion, const at::Tensor& image, float* partial) {
  if (fusion.scale == nullptr) {
    return {};
  }
  return {fusion.scale->data_ptr<float>(),
          fusion.shift->data_ptr<float>(),
          fusion.center != nullptr ? fusion.center->data_ptr<float>() : nullptr,
          partial,
          static_cast<int>(image.size(2)),
          static_cast<int>(image.size(3))};
}

// Launches grouped_conv_s1_kernel with TH-row tiles, persistent blocks per 72-channel chunk.
template <int TH, bool TRANSPOSE, bool FUSED>
void launch_s1(const at::Tensor& source, const at::Tensor& weight, at::Tensor& target, const Fusion& fusion) {
  constexpr int WPG = 2;
  constexpr int STAGES = kStages;
  using L = S1Layout<TH, STAGES>;
  const int threads = 32 * kGroupsPerBlock * WPG;
  const int batch = static_cast<int>(source.size(0));
  const int channels = static_cast<int>(source.size(1));
  const int chunks = channels / kBlockChannels;
  const int64_t tiles = batch * ((target.size(2) + TH - 1) / TH) * ((target.size(3) + 7) / 8);
  const at::Tensor packed = pack_weights<TRANSPOSE>(weight);
  const CUtensorMap source_tiles = source_map<0, TH>(source);
  const CUtensorMap target_tiles = tensor_map(target, 8, TH);
  const CUtensorMap y_tiles = FUSED && TRANSPOSE ? tensor_map(*fusion.y, 8, TH) : target_tiles;
  auto kernel = grouped_conv_s1_kernel<TH, WPG, STAGES, TRANSPOSE, FUSED>;
  const int resident = resident_blocks(kernel, threads, L::kBytes);
  const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(tiles, resident / chunks)));
  at::Tensor partial;
  if (FUSED) {
    partial = at::empty({blocks, 2, channels}, source.options().dtype(at::kFloat));
  }
  kernel<<<dim3(blocks, chunks), threads, L::kBytes, at::cuda::getCurrentCUDAStream()>>>(
      source_tiles, target_tiles, y_tiles, reinterpret_cast<const uint4*>(packed.data_ptr<at::BFloat16>()), batch,
      static_cast<int>(source.size(2)), static_cast<int>(source.size(3)), channels,
      bn_args(fusion, source, FUSED ? partial.data_ptr<float>() : nullptr));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (FUSED) {
    *fusion.sums = block_sums(partial);
  }
}

// The stride-1 convolution (or with TRANSPOSE its input gradient) at the tile height that pads the rows least; the
// fused kernels leave out 20-row tiles, which spill registers there.
template <bool TRANSPOSE, bool FUSED>
void launch_stride_one(const at::Tensor& source, const at::Tensor& weight, at::Tensor& target, const Fusion& fusion) {
  const int height = FUSED ? tile_height(target.size(2), {8, 12, 16}) : tile_height(target.size(2), {8, 12, 16, 20});
  switch (height) {
    case 8:
      return launch_s1<8, TRANSPOSE, FUSED>(source, weight, target, fusion);
    case 12:
      return launch_s1<12, TRANSPOSE, FUSED>(source, weight, target, fusion);
    case 16:
      return launch_s1<16, TRANSPOSE, FUSED>(source, weight, target, fusion);
    default:
      return launch_s1<20, TRANSPOSE, FUSED>(source, weight, target, fusion);
  }
}

// MODE 1: the stride-2 convolution of source into target; MODE 2: its input gradient, source the output gradient.
template <int MODE, int TH, bool FUSED>
void launch_s2(const at::Tensor& source, const at::Tensor& weight, at::Tensor& target, const Fusion& fusion) {
  constexpr int WPG = 2;
  constexpr int STAGES = kStages;
  using L = S2Layout<MODE, TH, STAGES>;
  const int threads = 32 * kGroupsPerBlock * WPG;
  const int batch = static_cast<int>(source.size(0));
  const int channels = static_cast<int>(source.size(1));
  const int chunks = channels / kBlockChannels;
  const int64_t cells_h = MODE == 2 ? (target.size(2) + 1) / 2 : target.size(2);
  const int64_t cells_w = MODE == 2 ? (target.size(3) + 1) / 2 : target.size(3);
  const int64_t tiles = batch * ((cells_h + TH - 1) / TH) * ((cells_w + 7) / 8);
  const CUtensorMap source_tiles = source_map<MODE, TH>(source);
  const CUtensorMap target_tiles = tensor_map(target, MODE == 2 ? 16 : 8, MODE == 2 ? 2 * TH : TH);
  const CUtensorMap y_tiles = FUSED && MODE == 2 ? tensor_map(*fusion.y, 16, 2 * TH) : target_tiles;
  auto run = [&](auto kernel, const auto* weight_data) {
    const int resident = resident_blocks(kernel, threads, L::kBytes);
    const int blocks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(tiles, resident / chunks)));
    at::Tensor partial;
    if (FUSED) {
      partial = at::empty({blocks, 2, channels}, source.options().dtype(at::kFloat));
    }
    kernel<<<dim3(blocks, chunks), threads, L::kBytes, at::cuda::getCurrentCUDAStream()>>>(
        source_tiles, target_tiles, y_tiles, weight_data, strides_of(weight),
        reinterpret_cast<__nv_bfloat16*>(target.data_ptr<at::BFloat16>()), static_cast<int>(target.size(2)),
        static_cast<int>(target.size(3)), channels, batch, static_cast<int>(cells_h), static_cast<int>(cells_w),
        bn_args(fusion, source, FUSED ? partial.data_ptr<float>() : nullptr));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (FUSED) {
      *fusion.sums = block_sums(partial);
    }
  };
  if (weight.scalar_type() == at::kFloat) {
    run(grouped_conv_s2_kernel<MODE, TH, WPG, STAGES, float, FUSED>, weight.data_ptr<float>());
  } else {
    run(grouped_conv_s2_kernel<MODE, TH, WPG, STAGES, __nv_bfloat16, FUSED>,
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()));
  }
}

// Forward op: channels-last bf16 x [N, C, H, W], C a multiple of 72, and a [C, 24, 3, 3] fp32 or bf16 weight.
at::Tensor regnet_grouped_conv(const at::Tensor& x, const at::Tensor& weight, int64_t stride) {
  check_inputs(x, weight);
  TORCH_CHECK(stride == 1 || stride == 2, "regnet_grouped_conv: stride must be 1 or 2");
  const c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty({x.size(0), x.size(1), output_size(x.size(2), stride), output_size(x.size(3), stride)},
                     x.options().memory_format(at::MemoryFormat::ChannelsLast));
  if (y.numel() == 0) {
    return y;
  }
  if (stride == 1) {
    launch_stride_one<false, false>(x, weight, y, {});
  } else {
    launch_s2<1, 4, false>(x, weight, y, {});
  }
  return y;
}

// Launches grouped_conv_weight_grad_kernel in clusters of kGradCluster blocks and reduces the clusters' partials into
// grad_weight.
template <int MODE, int TH, bool FUSED>
void launch_weight_grad(const at::Tensor& x, const at::Tensor& grad, at::Tensor& grad_weight, const Fusion& fusion) {
  constexpr int WPG = 2;
  constexpr int STAGES = kStages;
  using L = GradLayout<MODE, TH, STAGES>;
  const int threads = 32 * kGroupsPerBlock * WPG;
  const int batch = static_cast<int>(x.size(0));
  const int channels = static_cast<int>(x.size(1));
  const int64_t tiles = batch * ((grad.size(2) + TH - 1) / TH) * ((grad.size(3) + 7) / 8);
  const int chunks = channels / kBlockChannels;
  auto kernel = grouped_conv_weight_grad_kernel<MODE, TH, WPG, STAGES, FUSED>;
  const int bytes = std::max(L::kBytes, kGradBlockSumBytes);
  const int resident = resident_blocks(kernel, threads, bytes);
  int64_t blocks = std::max<int64_t>(1, std::min<int64_t>(tiles, resident / chunks));
  const int64_t per_block = (tiles + blocks - 1) / blocks;
  blocks = (tiles + per_block - 1) / per_block;
  // Whole clusters; blocks past the tiles add zeros.
  blocks = (blocks + kGradCluster - 1) / kGradCluster * kGradCluster;
  const int64_t clusters = blocks / kGradCluster;
  const int64_t rows = channels * kGradRows;
  auto partial = at::empty({clusters, rows}, x.options().dtype(at::kFloat));
  const CUtensorMap x_tiles = source_map<MODE, TH>(x);
  const CUtensorMap grad_tiles = tensor_map(grad, 8, TH);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(chunks));
  config.blockDim = dim3(threads);
  config.dynamicSmemBytes = static_cast<size_t>(bytes);
  config.stream = stream;
  cudaLaunchAttribute attribute;
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = kGradCluster;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  config.attrs = &attribute;
  config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, x_tiles, grad_tiles, partial.data_ptr<float>(), batch,
                                    static_cast<int>(grad.size(2)), static_cast<int>(grad.size(3)), channels,
                                    bn_args(fusion, x, nullptr)));
  const dim3 reduce_threads(32, 8);
  const int reduce_blocks = static_cast<int>((rows + 31) / 32);
  if (grad_weight.scalar_type() == at::kFloat) {
    grouped_conv_weight_grad_reduce_kernel<float><<<reduce_blocks, reduce_threads, 0, stream>>>(
        partial.data_ptr<float>(), grad_weight.data_ptr<float>(), strides_of(grad_weight), static_cast<int>(clusters),
        static_cast<int>(rows));
  } else {
    grouped_conv_weight_grad_reduce_kernel<__nv_bfloat16><<<reduce_blocks, reduce_threads, 0, stream>>>(
        partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(grad_weight.data_ptr<at::BFloat16>()),
        strides_of(grad_weight), static_cast<int>(clusters), static_cast<int>(rows));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The weight gradient [C, 24, 3, 3] in grad_weight's dtype, at the tile height that pads the rows least.
template <bool FUSED>
void weight_grad(const at::Tensor& x, const at::Tensor& grad, at::Tensor& grad_weight, int64_t stride,
                 const Fusion& fusion) {
  const int64_t rows = grad.size(2);
  if (stride == 1) {
    switch (tile_height(rows, {8, 12, 16, 20})) {
      case 8:
        return launch_weight_grad<0, 8, FUSED>(x, grad, grad_weight, fusion);
      case 12:
        return launch_weight_grad<0, 12, FUSED>(x, grad, grad_weight, fusion);
      case 16:
        return launch_weight_grad<0, 16, FUSED>(x, grad, grad_weight, fusion);
      default:
        return launch_weight_grad<0, 20, FUSED>(x, grad, grad_weight, fusion);
    }
  }
  if (tile_height(rows, {4, 8}) == 4) {
    return launch_weight_grad<1, 4, FUSED>(x, grad, grad_weight, fusion);
  }
  return launch_weight_grad<1, 8, FUSED>(x, grad, grad_weight, fusion);
}

at::Tensor checked_grad(const at::Tensor& grad_in, const at::Tensor& x, int64_t stride) {
  TORCH_CHECK(stride == 1 || stride == 2, "regnet_grouped_conv: stride must be 1 or 2");
  const auto grad = grad_in.to(at::kBFloat16).contiguous(at::MemoryFormat::ChannelsLast);
  TORCH_CHECK(grad.size(0) == x.size(0) && grad.size(1) == x.size(1) &&
                  grad.size(2) == output_size(x.size(2), stride) && grad.size(3) == output_size(x.size(3), stride),
              "regnet_grouped_conv: bad grad shape");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0, "regnet_grouped_conv: grad must be aligned");
  return grad;
}

// Backward op: the gradients of x (channels-last bf16) and weight (in weight's dtype and strides).
std::tuple<at::Tensor, at::Tensor> regnet_grouped_conv_backward(const at::Tensor& grad_in, const at::Tensor& x,
                                                                const at::Tensor& weight, int64_t stride) {
  check_inputs(x, weight);
  const c10::cuda::CUDAGuard guard(x.device());
  const auto grad = checked_grad(grad_in, x, stride);
  auto grad_x = at::empty_like(x, at::MemoryFormat::ChannelsLast);
  auto grad_weight = at::empty_like(weight);
  if (x.numel() == 0) {
    grad_weight.zero_();
    return {grad_x, grad_weight};
  }
  if (stride == 1) {
    launch_stride_one<true, false>(grad, weight, grad_x, {});
  } else {
    launch_s2<2, 4, false>(grad, weight, grad_x, {});
  }
  weight_grad<false>(x, grad, grad_weight, stride, {});
  return {grad_x, grad_weight};
}

void check_channel_vector(const at::Tensor& v, const at::Tensor& x, const char* name) {
  TORCH_CHECK(v.is_cuda() && v.device() == x.device() && v.scalar_type() == at::kFloat && v.dim() == 1 &&
                  v.size(0) == x.size(1) && v.is_contiguous(),
              "regnet_bn_relu_grouped_conv: ", name, " must be a contiguous fp32 [C] tensor on x's device");
}

// Fused forward op: the convolution of relu(y * scale + shift) and its [2, C] per-channel sums of the output and its
// square. center only matters to the backward; it is an input here so autograd can hand it over.
std::tuple<at::Tensor, at::Tensor> regnet_bn_relu_grouped_conv(const at::Tensor& y, const at::Tensor& scale,
                                                               const at::Tensor& shift, const at::Tensor& /*center*/,
                                                               const at::Tensor& weight, int64_t stride) {
  check_inputs(y, weight);
  check_channel_vector(scale, y, "scale");
  check_channel_vector(shift, y, "shift");
  TORCH_CHECK(stride == 1 || stride == 2, "regnet_grouped_conv: stride must be 1 or 2");
  const c10::cuda::CUDAGuard guard(y.device());
  auto out = at::empty({y.size(0), y.size(1), output_size(y.size(2), stride), output_size(y.size(3), stride)},
                       y.options().memory_format(at::MemoryFormat::ChannelsLast));
  auto sums = at::zeros({2, y.size(1)}, y.options().dtype(at::kFloat));
  if (out.numel() == 0) {
    return {out, sums};
  }
  Fusion fusion;
  fusion.scale = &scale;
  fusion.shift = &shift;
  fusion.sums = &sums;
  if (stride == 1) {
    launch_stride_one<false, true>(y, weight, out, fusion);
  } else {
    launch_s2<1, 4, true>(y, weight, out, fusion);
  }
  return {out, sums};
}

// Fused backward op: the gradients of y and weight, and [2, C] sums of the gradients of shift and of scale (the latter
// around center).
std::tuple<at::Tensor, at::Tensor, at::Tensor> regnet_bn_relu_grouped_conv_backward(
    const at::Tensor& grad_in, const at::Tensor& y, const at::Tensor& scale, const at::Tensor& shift,
    const at::Tensor& center, const at::Tensor& weight, int64_t stride) {
  check_inputs(y, weight);
  check_channel_vector(scale, y, "scale");
  check_channel_vector(shift, y, "shift");
  check_channel_vector(center, y, "center");
  const c10::cuda::CUDAGuard guard(y.device());
  const auto grad = checked_grad(grad_in, y, stride);
  auto grad_y = at::empty_like(y, at::MemoryFormat::ChannelsLast);
  auto grad_weight = at::empty_like(weight);
  auto sums = at::zeros({2, y.size(1)}, y.options().dtype(at::kFloat));
  if (y.numel() == 0) {
    grad_weight.zero_();
    return {grad_y, grad_weight, sums};
  }
  Fusion fusion;
  fusion.scale = &scale;
  fusion.shift = &shift;
  fusion.center = &center;
  fusion.y = &y;
  fusion.sums = &sums;
  if (stride == 1) {
    launch_stride_one<true, true>(grad, weight, grad_y, fusion);
  } else {
    launch_s2<2, 4, true>(grad, weight, grad_y, fusion);
  }
  weight_grad<true>(y, grad, grad_weight, stride, fusion);
  return {grad_y, grad_weight, sums};
}

}  // namespace

TORCH_LIBRARY_IMPL(py123d_garage_cuda, CUDA, m) {
  m.impl("regnet_grouped_conv", &regnet_grouped_conv);
  m.impl("regnet_grouped_conv_backward", &regnet_grouped_conv_backward);
  m.impl("regnet_bn_relu_grouped_conv", &regnet_bn_relu_grouped_conv);
  m.impl("regnet_bn_relu_grouped_conv_backward", &regnet_bn_relu_grouped_conv_backward);
}

}  // namespace py123d_garage_cuda
