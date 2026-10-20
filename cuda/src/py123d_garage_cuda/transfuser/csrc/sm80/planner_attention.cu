// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// The flow planner's attention, out = dropout(softmax(LN_q(q) LN_k(k)^T / sqrt(32), mask)) v per head, for head dim 32
// and at most 256 keys, in bf16 with fp32 accumulation.
//
// It rounds to bf16 where the compiled torch reference does: the normalized queries and keys, the probabilities, and
// the gradients of the normalized queries and keys. q is read at columns [q_col + 32 h, +32) of q_src rows, k and v at
// k_col and v_col of kv_src rows; self-attention passes one fused QKV tensor as both.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include <cstdint>
#include <type_traits>

#include "transfuser/csrc/sm80/planner.cuh"
#include "transfuser/csrc/sm80/ptx.cuh"

namespace py123d_garage_cuda {
namespace {

using namespace planner;

constexpr int kHeadDim = 32;
// Forward blocks: 4 warps of three 16-query tiles each, 4 blocks per SM.
constexpr int kForwardThreads = 128;
// Shared-memory row pitch in bf16: 80 bytes puts 8 consecutive rows' 16-byte chunks in distinct bank groups.
constexpr int kPitch = kHeadDim + 8;
// Keys per softmax chunk, as 8-key accumulator tiles.
constexpr int kChunk = 64;
constexpr int kChunkTiles = kChunk / 8;
constexpr int kMaxKeys = 256;
constexpr float kLog2e = 1.4426950408889634f;
// Backward warps own 32 keys each, in query groups of at most kGroupTiles 16-query tiles.
constexpr int kKeysPerWarp = 32;
constexpr int kMaxBackwardWarps = 12;
constexpr int kGroupTiles = 12;

struct Params {
  const __nv_bfloat16* q_src;
  const __nv_bfloat16* kv_src;
  int64_t q_batch_stride, q_row_stride, kv_batch_stride, kv_row_stride;
  int q_col, k_col, v_col;
  int batch, heads, queries, keys;
  int q_tiles, chunks;  // 16-query tiles, 64-key chunks
  const float* q_weight;
  const float* q_bias;
  const float* k_weight;
  const float* k_bias;
  const bool* mask;  // [queries, keys], true where a query may attend; nullptr attends everywhere
  // The mask as bits, [q_tiles * 16, 2 chunks] words (key k of a row at bit k % 32 of word k / 32; keys past the end
  // and padding rows clear), and per (query tile, chunk) whether any bit is set; only with a mask.
  const uint32_t* mask_words;
  const unsigned char* tile_attended;
  int forward_tiles;  // query tiles per forward block
  float eps;
  float scale_log2;         // 1/sqrt(32) * log2(e)
  uint32_t drop_threshold;  // 16-bit; 0 keeps everything
  float keep_scale;         // 1 / (1 - p)
  at::PhiloxCudaState philox;
  // [batch * heads, q_tiles, chunks, 32 lanes]: per lane and chunk, bit 4 t + i of 8-key tile t holds the keep
  // decision of its i-th accumulator element (rows a, a, b, b; columns c, c + 1, c, c + 1).
  uint32_t* keep_bits;
  __nv_bfloat16* out;  // [batch, queries, heads * 32]
  float* lse;          // [batch, heads, queries], log2 domain of the scaled scores
  // Backward.
  const __nv_bfloat16* grad;  // like out
  __nv_bfloat16* dq_src;      // like q_src, written at the q columns
  __nv_bfloat16* dkv_src;     // like kv_src, written at the k and v columns
  int64_t dq_batch_stride, dq_row_stride, dkv_batch_stride, dkv_row_stride;
  float* q_norm_partial;  // [batch * heads, 64]: weight gradient, then bias gradient
  float* k_norm_partial;
};

__host__ __device__ __forceinline__ int round_up16(int value) { return (value + 15) / 16 * 16; }

// Stores four 8 x 8 bf16 matrices transposed: lane l gives the address of stored row l % 8 of matrix l / 8, which is
// column l % 8 of the register matrix r[l / 8] (thread T holding row T / 4, columns 2 (T % 4) and 2 (T % 4) + 1).
__device__ __forceinline__ void stmatrix_x4_trans(void* pointer, const uint32_t (&r)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(shared_address(pointer)),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3])
      : "memory");
#else
  const int lane = threadIdx.x & 31;
  const uint32_t address = shared_address(pointer);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const uint32_t row = __shfl_sync(0xffffffffu, address, 8 * i + 2 * (lane & 3) + half);
      const uint16_t value = static_cast<uint16_t>(half ? r[i] >> 16 : r[i] & 0xffffu);
      asm volatile("st.shared.u16 [%0], %1;\n" ::"r"(row + (lane >> 2) * 2), "h"(value) : "memory");
    }
  }
#endif
}

// The A fragment (16 rows x 16 columns) of rows [row0, +16) and columns [col0, +16) of a tile with the given pitch.
__device__ __forceinline__ void load_a(uint32_t (&a)[4], const __nv_bfloat16* tile, int row0, int col0, int lane,
                                       int pitch = kPitch) {
  ldmatrix_x4(a, tile + (row0 + (lane & 15)) * pitch + col0 + (lane >> 4) * 8);
}

// B fragments of b = tile^T for the 8 tile rows [row0, +8) (b's columns) over all 32 tile columns (b's k):
// r[0..1] for k in [0, 16), r[2..3] for k in [16, 32).
__device__ __forceinline__ void load_b_rows(uint32_t (&r)[4], const __nv_bfloat16* tile, int row0, int lane) {
  ldmatrix_x4(r, tile + (row0 + (lane & 7)) * kPitch + (lane >> 3) * 8);
}

// B fragments of b = tile for tile rows [row0, +16) (b's k) and tile columns [col0, +16) (b's columns):
// r[0..1] for columns [col0, +8), r[2..3] for [col0 + 8, +8).
__device__ __forceinline__ void load_b_trans(uint32_t (&r)[4], const __nv_bfloat16* tile, int row0, int col0,
                                             int lane) {
  ldmatrix_x4_trans(r, tile + (row0 + (lane & 7) + ((lane >> 3) & 1) * 8) * kPitch + col0 + (lane >> 4) * 8);
}

// The A fragment of accumulator tiles t0 (columns [0, 8)) and t1 ([8, 16)), rounded to bf16.
__device__ __forceinline__ void accumulators_to_a(uint32_t (&a)[4], const float (&t0)[4], const float (&t1)[4]) {
  a[0] = pack_bf16(t0[0], t0[1]);
  a[1] = pack_bf16(t0[2], t0[3]);
  a[2] = pack_bf16(t1[0], t1[1]);
  a[3] = pack_bf16(t1[2], t1[3]);
}

// Mean and reciprocal standard deviation of a 32-value row held as 8 values by each of 4 consecutive lanes.
__device__ __forceinline__ float2 row_stats(const float (&x)[8], float eps) {
  float sum = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) sum += x[i];
  sum += __shfl_xor_sync(0xffffffffu, sum, 1);
  sum += __shfl_xor_sync(0xffffffffu, sum, 2);
  const float mean = sum * (1.f / kHeadDim);
  float squares = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) squares += (x[i] - mean) * (x[i] - mean);
  squares += __shfl_xor_sync(0xffffffffu, squares, 1);
  squares += __shfl_xor_sync(0xffffffffu, squares, 2);
  return make_float2(mean, rsqrtf(squares * (1.f / kHeadDim) + eps));
}

// Loads per thread in flight at once while staging.
constexpr int kStageBatch = 4;

// Rows [row0, row0 + count) of one head, LayerNorm'd and rounded to bf16, into a pitch-kPitch tile, 4 threads per row;
// rows past `valid` become zeros, and with stats their (mean, rstd) go to stats[row - row0]. count must be a multiple
// of 8 so that a warp's 8 rows of a pass are all in or all out and its shuffles stay converged.
__device__ void stage_normalized(__nv_bfloat16* tile, const __nv_bfloat16* src, int64_t row_stride, int col, int row0,
                                 int count, int valid, const float* weight, const float* bias, float eps, float2* stats,
                                 int thread, int threads) {
  const int part = thread & 3, rows_per_pass = threads / 4;
  float w[8], b[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    w[i] = weight[part * 8 + i];
    b[i] = bias[part * 8 + i];
  }
  for (int base = thread >> 2; base < count; base += kStageBatch * rows_per_pass) {
    uint4 raw[kStageBatch];
#pragma unroll
    for (int batch = 0; batch < kStageBatch; ++batch) {
      const int local = base + batch * rows_per_pass, row = row0 + local;
      raw[batch] = local < count && row < valid
                       ? *reinterpret_cast<const uint4*>(src + row * row_stride + col + part * 8)
                       : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int batch = 0; batch < kStageBatch; ++batch) {
      const int local = base + batch * rows_per_pass, row = row0 + local;
      if (local >= count) break;
      float x[8];
      unpack8(raw[batch], x);
      const float2 moments = row_stats(x, eps);
      float y[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) y[i] = row < valid ? (x[i] - moments.x) * moments.y * w[i] + b[i] : 0.f;
      *reinterpret_cast<uint4*>(tile + local * kPitch + part * 8) = pack8(y);
      if (stats != nullptr && part == 0) stats[local] = moments;
    }
  }
}

// Rows [row0, row0 + count) of one head copied into a pitch-kPitch tile; rows past `valid` become zeros.
__device__ void stage_plain(__nv_bfloat16* tile, const __nv_bfloat16* src, int64_t row_stride, int col, int row0,
                            int count, int valid, int thread, int threads) {
  for (int base = thread; base < count * 4; base += kStageBatch * threads) {
    uint4 raw[kStageBatch];
#pragma unroll
    for (int batch = 0; batch < kStageBatch; ++batch) {
      const int index = base + batch * threads, local = index >> 2, row = row0 + local;
      raw[batch] = index < count * 4 && row < valid
                       ? *reinterpret_cast<const uint4*>(src + row * row_stride + col + (index & 3) * 8)
                       : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int batch = 0; batch < kStageBatch; ++batch) {
      const int index = base + batch * threads;
      if (index < count * 4) *reinterpret_cast<uint4*>(tile + (index >> 2) * kPitch + (index & 3) * 8) = raw[batch];
    }
  }
}

// Rows [0, count) of a raw staged tile LayerNorm'd in place and rounded to bf16, 4 threads per row; rows past `valid`
// become zeros, and every row's (mean, rstd) goes to stats.
__device__ void normalize_in_place(__nv_bfloat16* tile, int count, int valid, const float* weight, const float* bias,
                                   float eps, float2* stats, int thread, int threads) {
  const int part = thread & 3, rows_per_pass = threads / 4;
  float w[8], b[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    w[i] = weight[part * 8 + i];
    b[i] = bias[part * 8 + i];
  }
  for (int base = thread >> 2; base < count; base += kStageBatch * rows_per_pass) {
    uint4 raw[kStageBatch];
    // Branch-free over the batch, so the rows' reductions interleave; rows past count are computed and dropped.
#pragma unroll
    for (int batch = 0; batch < kStageBatch; ++batch) {
      const int row = base + batch * rows_per_pass;
      raw[batch] =
          row < count ? *reinterpret_cast<const uint4*>(tile + row * kPitch + part * 8) : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int batch = 0; batch < kStageBatch; ++batch) {
      const int row = base + batch * rows_per_pass;
      float x[8], y[8];
      unpack8(raw[batch], x);
      const float2 moments = row_stats(x, eps);
#pragma unroll
      for (int i = 0; i < 8; ++i) y[i] = row < valid ? (x[i] - moments.x) * moments.y * w[i] + b[i] : 0.f;
      if (row < count) {
        *reinterpret_cast<uint4*>(tile + row * kPitch + part * 8) = pack8(y);
        if (part == 0) stats[row] = moments;
      }
    }
  }
}

// Rows [0, count) of one head copied asynchronously into a pitch-kPitch tile by the whole block; rows past `valid`
// become zeros.
__device__ __forceinline__ void copy_rows_async(__nv_bfloat16* tile, const __nv_bfloat16* src, int64_t row_stride,
                                                int col, int count, int valid) {
  for (int index = threadIdx.x; index < count * 4; index += blockDim.x) {
    const int row = index >> 2, part = index & 3;
    const bool in_range = row < valid;
    cp_async_16(tile + row * kPitch + part * 8, src + (in_range ? row * row_stride + col + part * 8 : 0), in_range);
  }
}

// count 32-bit words (a multiple of 4, 16-byte aligned) copied asynchronously by the whole block.
__device__ __forceinline__ void copy_words_async(uint32_t* target, const uint32_t* source, int count) {
  for (int index = threadIdx.x * 4; index < count; index += blockDim.x * 4) {
    cp_async_16(target + index, source + index, true);
  }
}

// The attend bits of keys [64 chunk, +64) for a query row (padding rows have none).
__device__ __forceinline__ uint64_t chunk_mask(const Params& p, int query, int chunk) {
  const uint2 words = *reinterpret_cast<const uint2*>(p.mask_words + query * 2 * p.chunks + 2 * chunk);
  return (static_cast<uint64_t>(words.y) << 32) | words.x;
}

__device__ __forceinline__ bool mask_bit(uint64_t bits, int key_in_chunk) { return (bits >> key_in_chunk) & 1u; }

// One block per 16-query tile, one thread per (row, 32-key word): the mask as bits and per chunk whether the tile may
// attend to any of its keys.
__global__ void mask_words_kernel(const bool* mask, int queries, int keys, int chunks, uint32_t* words,
                                  unsigned char* tile_attended) {
  __shared__ uint32_t tile_words[16 * 2 * kMaxKeys / kChunk];
  const int row_words = 2 * chunks;
  const int row = threadIdx.x / row_words, word_index = threadIdx.x % row_words;
  const int query = blockIdx.x * 16 + row;
  uint32_t word = 0u;
  if (query < queries) {
    for (int bit = 0; bit < 32; ++bit) {
      const int key = word_index * 32 + bit;
      if (key < keys && mask[static_cast<int64_t>(query) * keys + key]) word |= 1u << bit;
    }
  }
  words[static_cast<int64_t>(query) * row_words + word_index] = word;
  tile_words[threadIdx.x] = word;
  __syncthreads();
  if (threadIdx.x < chunks) {
    uint32_t any = 0u;
    for (int r = 0; r < 16; ++r)
      any |= tile_words[r * row_words + 2 * threadIdx.x] | tile_words[r * row_words + 2 * threadIdx.x + 1];
    tile_attended[blockIdx.x * chunks + threadIdx.x] = any != 0u;
  }
}

// A warp's raw rows of one 16-query tile: rows lane / 4 and lane / 4 + 8, 8 values from column 8 (lane % 4).
__device__ __forceinline__ void load_query_rows(uint4 (&raw)[2], const Params& p, const __nv_bfloat16* q_rows, int head,
                                                int query0, int lane) {
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = query0 + (lane >> 2) + 8 * r;
    raw[r] = row < p.queries ? *reinterpret_cast<const uint4*>(q_rows + row * p.q_row_stride + p.q_col +
                                                               head * kHeadDim + (lane & 3) * 8)
                             : make_uint4(0u, 0u, 0u, 0u);
  }
}

// The rows of load_query_rows, LayerNorm'd and rounded to bf16, into the warp's 16-row tile.
__device__ __forceinline__ void normalize_query_rows(__nv_bfloat16* stage, const uint4 (&raw)[2], const Params& p,
                                                     int query0, int lane) {
  const int part = lane & 3;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int local = (lane >> 2) + 8 * r;
    float x[8], y[8];
    unpack8(raw[r], x);
    const float2 moments = row_stats(x, p.eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      y[i] = query0 + local < p.queries
                 ? (x[i] - moments.x) * moments.y * p.q_weight[part * 8 + i] + p.q_bias[part * 8 + i]
                 : 0.f;
    }
    *reinterpret_cast<uint4*>(stage + local * kPitch + part * 8) = pack8(y);
  }
}

// 2^x with denormal results flushed to zero, one MUFU instruction; the probabilities it feeds round to bf16.
__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// The ten round keys of a Philox stream, computed once per thread.
struct PhiloxKeys {
  uint32_t x[10], y[10];
};

__device__ __forceinline__ PhiloxKeys philox_keys(const PhiloxStream& stream) {
  PhiloxKeys keys;
#pragma unroll
  for (int round = 0; round < 10; ++round) {
    keys.x[round] = stream.key.x + round * 0x9E3779B9u;
    keys.y[round] = stream.key.y + round * 0xBB67AE85u;
  }
  return keys;
}

// philox_draw with precomputed round keys.
__device__ __forceinline__ uint4 philox_draw(const PhiloxStream& stream, const PhiloxKeys& keys, uint64_t subsequence) {
  uint4 counter = make_uint4(stream.offset.x, stream.offset.y, static_cast<uint32_t>(subsequence),
                             static_cast<uint32_t>(subsequence >> 32));
#pragma unroll
  for (int round = 0; round < 10; ++round) {
    const uint64_t product0 = static_cast<uint64_t>(0xD2511F53u) * counter.x;
    const uint64_t product1 = static_cast<uint64_t>(0xCD9E8D57u) * counter.z;
    counter =
        make_uint4(static_cast<uint32_t>(product1 >> 32) ^ counter.y ^ keys.x[round], static_cast<uint32_t>(product1),
                   static_cast<uint32_t>(product0 >> 32) ^ counter.w ^ keys.y[round], static_cast<uint32_t>(product0));
  }
  return counter;
}

// keep_pair against the threshold and the threshold shifted to the high half: bit 0 the low half, bit 1 the high.
__device__ __forceinline__ uint32_t keep_pair(uint32_t word, uint32_t threshold, uint32_t high_threshold) {
  return ((word & 0xffffu) >= threshold ? 1u : 0u) | (word >= high_threshold ? 2u : 0u);
}

// The lane's keep bits of a chunk of kTiles 8-key tiles (bit 4 t + i of tile t, element i), from one draw per pair of
// tiles; with kTail, tiles from `valid_tiles` on hold no key and are not drawn.
template <int kTiles, bool kTail>
__device__ __forceinline__ uint32_t chunk_keep_word(const PhiloxStream& stream, const PhiloxKeys& keys,
                                                    uint32_t threshold, int64_t pair0, int lane, int valid_tiles) {
  const uint32_t high_threshold = threshold << 16;
  uint32_t word = 0u;
#pragma unroll
  for (int j = 0; j < kTiles / 2; ++j) {
    if (!kTail || 2 * j < valid_tiles) {
      const uint4 bits = philox_draw(stream, keys, (pair0 + j) * 32 + lane);
      const uint32_t even = keep_pair(bits.x, threshold, high_threshold) | keep_pair(bits.y, threshold, high_threshold)
                                                                               << 2;
      const uint32_t odd = keep_pair(bits.z, threshold, high_threshold) | keep_pair(bits.w, threshold, high_threshold)
                                                                              << 2;
      word |= (even | (!kTail || 2 * j + 1 < valid_tiles ? odd << 4 : 0u)) << (8 * j);
    }
  }
  return word;
}

// The online softmax state of a warp's 16-query tile: maxima in raw score units, row sums and the output.
struct SoftmaxRows {
  float max_a, max_b, sum_a, sum_b;
  float o[4][4];
};

// One chunk of kTiles 8-key tiles from key0 for a warp's query tile; with kTail, keys from p.keys on are masked.
template <int kTiles, bool kMask, bool kDropout, bool kTail>
__device__ __forceinline__ void forward_chunk(SoftmaxRows& rows, const Params& p, const uint32_t (&q_frag)[2][4],
                                              const __nv_bfloat16* k_tile, const __nv_bfloat16* v_tile,
                                              const PhiloxStream& stream, const PhiloxKeys& keys, int64_t word_index,
                                              int key0, int chunk, int query_a, int query_b, int lane) {
  const int column = (lane & 3) * 2;
  const float scale_log2 = p.scale_log2;
  uint32_t keep_word = 0xffffffffu;
  if (kDropout) {
    keep_word = chunk_keep_word<kTiles, kTail>(stream, keys, p.drop_threshold, word_index * (kChunkTiles / 2), lane,
                                               (p.keys - key0 + 7) / 8);
    p.keep_bits[word_index * 32 + lane] = keep_word;
  }
  float s[kTiles][4];
#pragma unroll
  for (int t = 0; t < kTiles; ++t) {
    s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.f;
    uint32_t b[4];
    load_b_rows(b, k_tile, key0 + t * 8, lane);
    mma(s[t], q_frag[0], b[0], b[1]);
    mma(s[t], q_frag[1], b[2], b[3]);
  }
  if (kMask) {
    const uint64_t mask_a = chunk_mask(p, query_a, chunk), mask_b = chunk_mask(p, query_b, chunk);
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int key_in_chunk = t * 8 + column + i;
        if (!mask_bit(mask_a, key_in_chunk)) s[t][i] = -INFINITY;
        if (!mask_bit(mask_b, key_in_chunk)) s[t][2 + i] = -INFINITY;
      }
    }
  } else if (kTail) {
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        if (key0 + t * 8 + column + i >= p.keys) s[t][i] = s[t][2 + i] = -INFINITY;
      }
    }
  }
  float chunk_max_a = -INFINITY, chunk_max_b = -INFINITY;
#pragma unroll
  for (int t = 0; t < kTiles; ++t) {
    chunk_max_a = fmaxf(chunk_max_a, fmaxf(s[t][0], s[t][1]));
    chunk_max_b = fmaxf(chunk_max_b, fmaxf(s[t][2], s[t][3]));
  }
#pragma unroll
  for (int offset = 1; offset <= 2; offset <<= 1) {
    chunk_max_a = fmaxf(chunk_max_a, __shfl_xor_sync(0xffffffffu, chunk_max_a, offset));
    chunk_max_b = fmaxf(chunk_max_b, __shfl_xor_sync(0xffffffffu, chunk_max_b, offset));
  }
  const float new_max_a = fmaxf(rows.max_a, chunk_max_a), new_max_b = fmaxf(rows.max_b, chunk_max_b);
  // Rows with nothing attended yet shift by 0 and stay at zero.
  const float shift_a = new_max_a == -INFINITY ? 0.f : new_max_a * scale_log2;
  const float shift_b = new_max_b == -INFINITY ? 0.f : new_max_b * scale_log2;
  const float rescale_a = fast_exp2(fmaf(rows.max_a, scale_log2, -shift_a));
  const float rescale_b = fast_exp2(fmaf(rows.max_b, scale_log2, -shift_b));
  rows.max_a = new_max_a;
  rows.max_b = new_max_b;
  rows.sum_a *= rescale_a;
  rows.sum_b *= rescale_b;
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    rows.o[n][0] *= rescale_a;
    rows.o[n][1] *= rescale_a;
    rows.o[n][2] *= rescale_b;
    rows.o[n][3] *= rescale_b;
  }
#pragma unroll
  for (int t = 0; t < kTiles; ++t) {
    s[t][0] = fast_exp2(fmaf(s[t][0], scale_log2, -shift_a));
    s[t][1] = fast_exp2(fmaf(s[t][1], scale_log2, -shift_a));
    s[t][2] = fast_exp2(fmaf(s[t][2], scale_log2, -shift_b));
    s[t][3] = fast_exp2(fmaf(s[t][3], scale_log2, -shift_b));
    rows.sum_a += s[t][0] + s[t][1];
    rows.sum_b += s[t][2] + s[t][3];
    if (kDropout) {
      const uint32_t keep = keep_word >> (4 * t);
      s[t][0] = keep & 1u ? s[t][0] : 0.f;
      s[t][1] = keep & 2u ? s[t][1] : 0.f;
      s[t][2] = keep & 4u ? s[t][2] : 0.f;
      s[t][3] = keep & 8u ? s[t][3] : 0.f;
    }
  }
#pragma unroll
  for (int j = 0; j < kTiles / 2; ++j) {
    uint32_t a[4];
    accumulators_to_a(a, s[2 * j], s[2 * j + 1]);
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      uint32_t b[4];
      load_b_trans(b, v_tile, key0 + j * 16, n * 16, lane);
      mma(rows.o[2 * n], a, b[0], b[1]);
      mma(rows.o[2 * n + 1], a, b[2], b[3]);
    }
  }
}

// The last chunk, whose staged keys (padded to 16) end before 64.
template <bool kMask, bool kDropout>
__device__ __forceinline__ void forward_tail_chunk(SoftmaxRows& rows, const Params& p, const uint32_t (&q_frag)[2][4],
                                                   const __nv_bfloat16* k_tile, const __nv_bfloat16* v_tile,
                                                   const PhiloxStream& stream, const PhiloxKeys& keys,
                                                   int64_t word_index, int key0, int chunk, int query_a, int query_b,
                                                   int lane) {
  switch ((round_up16(p.keys) - key0) / 8) {
    case 2:
      forward_chunk<2, kMask, kDropout, true>(rows, p, q_frag, k_tile, v_tile, stream, keys, word_index, key0, chunk,
                                              query_a, query_b, lane);
      break;
    case 4:
      forward_chunk<4, kMask, kDropout, true>(rows, p, q_frag, k_tile, v_tile, stream, keys, word_index, key0, chunk,
                                              query_a, query_b, lane);
      break;
    case 6:
      forward_chunk<6, kMask, kDropout, true>(rows, p, q_frag, k_tile, v_tile, stream, keys, word_index, key0, chunk,
                                              query_a, query_b, lane);
      break;
    default:
      forward_chunk<8, kMask, kDropout, true>(rows, p, q_frag, k_tile, v_tile, stream, keys, word_index, key0, chunk,
                                              query_a, query_b, lane);
  }
}

// One block per batch row and head, blockIdx.y splitting the query tiles; each warp runs 16-query tiles over 64-key
// chunks with an online softmax and writes out [batch, queries, heads * 32] and lse [batch, heads, queries].
template <bool kMask, bool kDropout>
__global__ void __launch_bounds__(kForwardThreads, 4) attention_forward_kernel(const Params p) {
  extern __shared__ __align__(16) unsigned char shared_raw[];
  const int keys_padded = round_up16(p.keys);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  __nv_bfloat16* k_tile = reinterpret_cast<__nv_bfloat16*>(shared_raw);
  __nv_bfloat16* v_tile = k_tile + keys_padded * kPitch;
  __nv_bfloat16* q_stage = v_tile + keys_padded * kPitch + warp * 16 * kPitch;

  const int batch_head = blockIdx.x, batch = batch_head / p.heads, head = batch_head % p.heads;
  const __nv_bfloat16* q_rows = p.q_src + batch * p.q_batch_stride;
  const __nv_bfloat16* kv_rows = p.kv_src + batch * p.kv_batch_stride;
  stage_normalized(k_tile, kv_rows, p.kv_row_stride, p.k_col + head * kHeadDim, 0, keys_padded, p.keys, p.k_weight,
                   p.k_bias, p.eps, nullptr, threadIdx.x, blockDim.x);
  stage_plain(v_tile, kv_rows, p.kv_row_stride, p.v_col + head * kHeadDim, 0, keys_padded, p.keys, threadIdx.x,
              blockDim.x);
  __syncthreads();

  PhiloxStream stream{};
  PhiloxKeys keys{};
  if (kDropout) {
    stream = philox_stream(p.philox, nullptr);
    keys = philox_keys(stream);
  }
  const int column = (lane & 3) * 2;
  const int full_chunks = p.keys / kChunk;
  const int64_t out_row_stride = static_cast<int64_t>(p.heads) * kHeadDim;
  __nv_bfloat16* out_rows = p.out + static_cast<int64_t>(batch) * p.queries * out_row_stride + head * kHeadDim;
  const int first_tile = blockIdx.y * p.forward_tiles, last_tile = min(p.q_tiles, first_tile + p.forward_tiles);
  // Each tile's raw query rows load while the previous tile computes.
  uint4 q_raw[2];
  if (first_tile + warp < last_tile) load_query_rows(q_raw, p, q_rows, head, (first_tile + warp) * 16, lane);
  for (int tile = first_tile + warp; tile < last_tile; tile += warps) {
    const int query0 = tile * 16;
    const int query_a = query0 + (lane >> 2), query_b = query_a + 8;
    __syncwarp();
    normalize_query_rows(q_stage, q_raw, p, query0, lane);
    __syncwarp();
    uint32_t q_frag[2][4];
    load_a(q_frag[0], q_stage, 0, 0, lane);
    load_a(q_frag[1], q_stage, 0, 16, lane);
    if (tile + warps < last_tile) load_query_rows(q_raw, p, q_rows, head, (tile + warps) * 16, lane);

    SoftmaxRows rows;
    rows.max_a = rows.max_b = -INFINITY;
    rows.sum_a = rows.sum_b = 0.f;
#pragma unroll
    for (int n = 0; n < 4; ++n) rows.o[n][0] = rows.o[n][1] = rows.o[n][2] = rows.o[n][3] = 0.f;
    const int64_t first_word = (static_cast<int64_t>(batch_head) * p.q_tiles + tile) * p.chunks;
    for (int chunk = 0; chunk < full_chunks; ++chunk) {
      if (kMask && !p.tile_attended[tile * p.chunks + chunk]) continue;
      forward_chunk<kChunkTiles, kMask, kDropout, false>(rows, p, q_frag, k_tile, v_tile, stream, keys,
                                                         first_word + chunk, chunk * kChunk, chunk, query_a, query_b,
                                                         lane);
    }
    if (full_chunks < p.chunks && !(kMask && !p.tile_attended[tile * p.chunks + full_chunks])) {
      forward_tail_chunk<kMask, kDropout>(rows, p, q_frag, k_tile, v_tile, stream, keys, first_word + full_chunks,
                                          full_chunks * kChunk, full_chunks, query_a, query_b, lane);
    }
    float sum_a = rows.sum_a, sum_b = rows.sum_b;
#pragma unroll
    for (int offset = 1; offset <= 2; offset <<= 1) {
      sum_a += __shfl_xor_sync(0xffffffffu, sum_a, offset);
      sum_b += __shfl_xor_sync(0xffffffffu, sum_b, offset);
    }
    const float norm_a = sum_a > 0.f ? p.keep_scale / sum_a : 0.f;
    const float norm_b = sum_b > 0.f ? p.keep_scale / sum_b : 0.f;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int dim = n * 8 + column;
      if (query_a < p.queries) {
        *reinterpret_cast<__nv_bfloat162*>(out_rows + query_a * out_row_stride + dim) =
            __floats2bfloat162_rn(rows.o[n][0] * norm_a, rows.o[n][1] * norm_a);
      }
      if (query_b < p.queries) {
        *reinterpret_cast<__nv_bfloat162*>(out_rows + query_b * out_row_stride + dim) =
            __floats2bfloat162_rn(rows.o[n][2] * norm_b, rows.o[n][3] * norm_b);
      }
    }
    if ((lane & 3) == 0) {
      float* lse_rows = p.lse + static_cast<int64_t>(batch_head) * p.queries;
      if (query_a < p.queries) lse_rows[query_a] = sum_a > 0.f ? rows.max_a * p.scale_log2 + log2f(sum_a) : INFINITY;
      if (query_b < p.queries) lse_rows[query_b] = sum_b > 0.f ? rows.max_b * p.scale_log2 + log2f(sum_b) : INFINITY;
    }
  }
}

// The bf16 pairs of one input row a thread's accumulator columns 8 n + column + {0, 1} cover.
__device__ __forceinline__ void load_raw_pairs(__nv_bfloat162 (&raw)[4], const __nv_bfloat16* row, int column) {
#pragma unroll
  for (int n = 0; n < 4; ++n) raw[n] = *reinterpret_cast<const __nv_bfloat162*>(row + n * 8 + column);
}

// Norm backward of one thread's 2 rows x 8 columns: grad holds the bf16-rounded gradient of the normalized values in
// accumulator layout (tile n: columns 8 n + 2 (lane % 4) + {0, 1}; [0..1] row a, [2..3] row b), raw the bf16 input
// pairs of rows a and b (load_raw_pairs), stats their (mean, rstd). Writes the input gradient and adds the weight and
// bias gradients of the thread's columns into weight_grad / bias_grad (index 2 n + i).
__device__ __forceinline__ void norm_backward(const float (&grad)[4][4], const __nv_bfloat162 (&raw_a)[4],
                                              const __nv_bfloat162 (&raw_b)[4], float2 stats_a, float2 stats_b,
                                              const float* weight, int column, float (&input_grad)[4][4],
                                              float (&weight_grad)[8], float (&bias_grad)[8]) {
  float normalized[4][4];
  float sum_a = 0.f, dot_a = 0.f, sum_b = 0.f, dot_b = 0.f;
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const int dim = n * 8 + column;
    const float2 xa = __bfloat1622float2(raw_a[n]);
    const float2 xb = __bfloat1622float2(raw_b[n]);
    normalized[n][0] = (xa.x - stats_a.x) * stats_a.y;
    normalized[n][1] = (xa.y - stats_a.x) * stats_a.y;
    normalized[n][2] = (xb.x - stats_b.x) * stats_b.y;
    normalized[n][3] = (xb.y - stats_b.x) * stats_b.y;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const float w = weight[dim + i];
      const float ga = grad[n][i] * w, gb = grad[n][2 + i] * w;
      sum_a += ga;
      dot_a += ga * normalized[n][i];
      sum_b += gb;
      dot_b += gb * normalized[n][2 + i];
      weight_grad[2 * n + i] += grad[n][i] * normalized[n][i] + grad[n][2 + i] * normalized[n][2 + i];
      bias_grad[2 * n + i] += grad[n][i] + grad[n][2 + i];
    }
  }
#pragma unroll
  for (int offset = 1; offset <= 2; offset <<= 1) {
    sum_a += __shfl_xor_sync(0xffffffffu, sum_a, offset);
    dot_a += __shfl_xor_sync(0xffffffffu, dot_a, offset);
    sum_b += __shfl_xor_sync(0xffffffffu, sum_b, offset);
    dot_b += __shfl_xor_sync(0xffffffffu, dot_b, offset);
  }
  const float mean_a = sum_a * (1.f / kHeadDim), mean_dot_a = dot_a * (1.f / kHeadDim);
  const float mean_b = sum_b * (1.f / kHeadDim), mean_dot_b = dot_b * (1.f / kHeadDim);
#pragma unroll
  for (int n = 0; n < 4; ++n) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const float w = weight[n * 8 + column + i];
      input_grad[n][i] = stats_a.y * (grad[n][i] * w - mean_a - normalized[n][i] * mean_dot_a);
      input_grad[n][2 + i] = stats_b.y * (grad[n][2 + i] * w - mean_b - normalized[n][2 + i] * mean_dot_b);
    }
  }
}

// Sums every thread's 8 column partials over the block into partial[0..31] (weight) and [32..63] (bias), in a fixed
// order. scratch holds warps * 64 floats.
__device__ void reduce_norm_partials(float (&weight_grad)[8], float (&bias_grad)[8], float* scratch, float* partial) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  // Lanes with equal lane % 4 hold the same columns.
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int offset = 4; offset <= 16; offset <<= 1) {
      weight_grad[i] += __shfl_xor_sync(0xffffffffu, weight_grad[i], offset);
      bias_grad[i] += __shfl_xor_sync(0xffffffffu, bias_grad[i], offset);
    }
  }
  __syncthreads();
  if (lane < 4) {
#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int dim = n * 8 + lane * 2 + i;
        scratch[warp * 64 + dim] = weight_grad[2 * n + i];
        scratch[warp * 64 + 32 + dim] = bias_grad[2 * n + i];
      }
    }
  }
  __syncthreads();
  for (int index = threadIdx.x; index < 64; index += blockDim.x) {
    float total = 0.f;
    for (int w = 0; w < warps; ++w) total += scratch[w * 64 + index];
    partial[index] = total;
  }
}

// Adds a warp's 8 column partials per thread (as norm_backward leaves them) to its row of 64 sums (weight, then bias).
__device__ __forceinline__ void add_norm_partials(float (&weight_grad)[8], float (&bias_grad)[8], float* row,
                                                  int lane) {
  // Lanes with equal lane % 4 hold the same columns.
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int offset = 4; offset <= 16; offset <<= 1) {
      weight_grad[i] += __shfl_xor_sync(0xffffffffu, weight_grad[i], offset);
      bias_grad[i] += __shfl_xor_sync(0xffffffffu, bias_grad[i], offset);
    }
  }
  if (lane < 4) {
#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int dim = n * 8 + lane * 2 + i;
        row[dim] += weight_grad[2 * n + i];
        row[32 + dim] += bias_grad[2 * n + i];
      }
    }
  }
}

// Writes the bf16 input gradient of two rows of one head at columns col + 32 head of rows (row_a, row_b) of target;
// rows past `valid` are skipped.
__device__ __forceinline__ void store_rows(__nv_bfloat16* target, int64_t row_stride, int row_a, int row_b, int valid,
                                           int column, const float (&values)[4][4]) {
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const int dim = n * 8 + column;
    if (row_a < valid) {
      *reinterpret_cast<__nv_bfloat162*>(target + row_a * row_stride + dim) =
          __floats2bfloat162_rn(values[n][0], values[n][1]);
    }
    if (row_b < valid) {
      *reinterpret_cast<__nv_bfloat162*>(target + row_b * row_stride + dim) =
          __floats2bfloat162_rn(values[n][2], values[n][3]);
    }
  }
}

__host__ __device__ __forceinline__ int min_int(int a, int b) { return a < b ? a : b; }

__host__ __device__ __forceinline__ int max_int(int a, int b) { return a > b ? a : b; }

// The backward's work split and shared memory: key_warps warps of 32 keys per query group of group_tiles tiles, and
// per group key_warps dS tiles of [16 queries, ds_pitch] bf16.
struct BackwardLayout {
  int keys_padded, queries_padded, key_tiles, key_warps, groups, group_tiles, ds_pitch;
  size_t k_tile, v_tile, q_tile, grad_tile, k_stats, q_stats, lse, delta, bits, mask_words, ds, scratch, total;
};

__host__ __device__ inline BackwardLayout backward_layout(int keys, int q_tiles, int chunks, bool dropout, bool mask) {
  BackwardLayout layout{};
  layout.keys_padded = round_up16(keys);
  layout.queries_padded = q_tiles * 16;
  layout.key_tiles = layout.keys_padded / 16;
  layout.key_warps = (layout.key_tiles + 1) / 2;
  layout.groups = max_int(1, min_int(kMaxBackwardWarps / layout.key_warps, q_tiles / kGroupTiles));
  layout.group_tiles = (q_tiles + layout.groups - 1) / layout.groups;
  layout.ds_pitch = layout.key_warps * kKeysPerWarp + 8;
  size_t offset = 0;
  auto take = [&offset](size_t bytes) {
    const size_t start = offset;
    offset = (offset + bytes + 15) / 16 * 16;
    return start;
  };
  layout.k_tile = take(layout.keys_padded * kPitch * sizeof(__nv_bfloat16));
  layout.v_tile = take(layout.keys_padded * kPitch * sizeof(__nv_bfloat16));
  // The query and gradient tiles are adjacent: after the main loop they hold the groups' key and value gradients.
  layout.q_tile = take(layout.queries_padded * kPitch * sizeof(__nv_bfloat16));
  layout.grad_tile = take(layout.queries_padded * kPitch * sizeof(__nv_bfloat16));
  layout.k_stats = take(layout.keys_padded * sizeof(float2));
  layout.q_stats = take(layout.queries_padded * sizeof(float2));
  layout.lse = take(layout.queries_padded * sizeof(float));
  layout.delta = take(layout.queries_padded * sizeof(float));
  layout.bits = take(dropout ? static_cast<size_t>(q_tiles) * chunks * 32 * sizeof(uint32_t) : 0);
  layout.mask_words = take(mask ? static_cast<size_t>(layout.queries_padded) * 2 * chunks * sizeof(uint32_t) : 0);
  layout.ds =
      take(static_cast<size_t>(layout.groups) * layout.key_warps * 16 * layout.ds_pitch * sizeof(__nv_bfloat16));
  // Per warp: its 64 query-norm sums, then the key-norm reduction's 64.
  layout.scratch = take(static_cast<size_t>(layout.groups) * layout.key_warps * 2 * 64 * sizeof(float));
  layout.total = offset;
  return layout;
}

// Floats of one warp's key and value gradients: 2 key tiles x 4 column tiles x 4 accumulators x 32 lanes, twice.
constexpr int kKeyGradientFloats = 2 * 2 * 4 * 4 * 32;

__device__ __forceinline__ void group_barrier(int group, int threads) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(group + 1), "r"(threads) : "memory");
}

// One block per batch row and head, its warps in query groups of key_warps warps. Within a group, warp k owns keys
// [32 k, +32) and walks the group's query tiles, keeping its key and value gradients in registers and storing its part
// of each tile's score gradient dS to shared memory; after every key_warps tiles the group's warps compute those tiles'
// query gradients from dS, one tile each. The groups' key and value gradients add up in group order at the end.
template <bool kMask, bool kDropout>
__global__ void __launch_bounds__(kMaxBackwardWarps * 32, 1) attention_backward_kernel(const Params p) {
  extern __shared__ __align__(16) unsigned char shared_raw[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const BackwardLayout layout = backward_layout(p.keys, p.q_tiles, p.chunks, kDropout, kMask);
  __nv_bfloat16* k_tile = reinterpret_cast<__nv_bfloat16*>(shared_raw + layout.k_tile);
  __nv_bfloat16* v_tile = reinterpret_cast<__nv_bfloat16*>(shared_raw + layout.v_tile);
  __nv_bfloat16* q_tile = reinterpret_cast<__nv_bfloat16*>(shared_raw + layout.q_tile);
  __nv_bfloat16* grad_tile = reinterpret_cast<__nv_bfloat16*>(shared_raw + layout.grad_tile);
  float2* k_stats = reinterpret_cast<float2*>(shared_raw + layout.k_stats);
  float2* q_stats = reinterpret_cast<float2*>(shared_raw + layout.q_stats);
  float* lse_rows = reinterpret_cast<float*>(shared_raw + layout.lse);
  float* delta_rows = reinterpret_cast<float*>(shared_raw + layout.delta);
  uint32_t* bits = reinterpret_cast<uint32_t*>(shared_raw + layout.bits);
  uint32_t* mask_words = reinterpret_cast<uint32_t*>(shared_raw + layout.mask_words);
  __nv_bfloat16* ds = reinterpret_cast<__nv_bfloat16*>(shared_raw + layout.ds);
  float* scratch = reinterpret_cast<float*>(shared_raw + layout.scratch);

  const int batch_head = blockIdx.x, batch = batch_head / p.heads, head = batch_head % p.heads;
  const __nv_bfloat16* q_rows = p.q_src + batch * p.q_batch_stride;
  const __nv_bfloat16* kv_rows = p.kv_src + batch * p.kv_batch_stride;
  const int64_t out_row_stride = static_cast<int64_t>(p.heads) * kHeadDim;
  const int64_t out_offset = static_cast<int64_t>(batch) * p.queries * out_row_stride + head * kHeadDim;
  // Every staged row goes to shared memory raw in one asynchronous batch, and the norms then run in place; the first
  // output rows of delta and their log-sum-exp load meanwhile.
  const int threads = blockDim.x;
  copy_rows_async(k_tile, kv_rows, p.kv_row_stride, p.k_col + head * kHeadDim, layout.keys_padded, p.keys);
  copy_rows_async(q_tile, q_rows, p.q_row_stride, p.q_col + head * kHeadDim, layout.queries_padded, p.queries);
  copy_rows_async(v_tile, kv_rows, p.kv_row_stride, p.v_col + head * kHeadDim, layout.keys_padded, p.keys);
  copy_rows_async(grad_tile, p.grad + out_offset, out_row_stride, 0, layout.queries_padded, p.queries);
  if (kDropout) {
    const int words = p.q_tiles * p.chunks * 32;
    copy_words_async(bits, p.keep_bits + static_cast<int64_t>(batch_head) * words, words);
  }
  if (kMask) copy_words_async(mask_words, p.mask_words, layout.queries_padded * 2 * p.chunks);
  cp_async_commit();
  float* q_sums = scratch;
  float* k_scratch = scratch + (threads >> 5) * 64;
  for (int index = threadIdx.x; index < (threads >> 5) * 64; index += threads) q_sums[index] = 0.f;
  const int part = threadIdx.x & 3, rows_per_pass = threads / 4;
  uint4 out_raw[kStageBatch];
  float lse[kStageBatch];
  // Rows base + batch * rows_per_pass of a pass of the delta loop, from global memory.
  auto load_out_rows = [&](int base) {
#pragma unroll
    for (int batch = 0; batch < kStageBatch; ++batch) {
      const int row = base + batch * rows_per_pass;
      const bool valid = row < p.queries;
      out_raw[batch] = valid ? *reinterpret_cast<const uint4*>(p.out + out_offset + row * out_row_stride + part * 8)
                             : make_uint4(0u, 0u, 0u, 0u);
      lse[batch] = valid ? p.lse[static_cast<int64_t>(batch_head) * p.queries + row] : INFINITY;
    }
  };
  load_out_rows(threadIdx.x >> 2);
  cp_async_wait<0>();
  __syncthreads();
  normalize_in_place(k_tile, layout.keys_padded, p.keys, p.k_weight, p.k_bias, p.eps, k_stats, threadIdx.x, threads);
  normalize_in_place(q_tile, layout.queries_padded, p.queries, p.q_weight, p.q_bias, p.eps, q_stats, threadIdx.x,
                     threads);
  // delta = rowsum(grad * out) from the staged gradient, and the log-sum-exp, 4 threads per row.
  for (int base = threadIdx.x >> 2; base < layout.queries_padded; base += kStageBatch * rows_per_pass) {
    if (base != (threadIdx.x >> 2)) load_out_rows(base);
#pragma unroll
    for (int batch = 0; batch < kStageBatch; ++batch) {
      const int row = base + batch * rows_per_pass;
      if (row >= layout.queries_padded) break;
      float g[8], o[8];
      unpack8(*reinterpret_cast<const uint4*>(grad_tile + row * kPitch + part * 8), g);
      unpack8(out_raw[batch], o);
      float dot = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) dot += g[i] * o[i];
      dot += __shfl_xor_sync(0xffffffffu, dot, 1);
      dot += __shfl_xor_sync(0xffffffffu, dot, 2);
      if (part == 0) {
        delta_rows[row] = dot;
        lse_rows[row] = lse[batch];
      }
    }
  }
  __syncthreads();

  const int key_warps = layout.key_warps, group = warp / key_warps, key_warp = warp - group * key_warps;
  const int group_threads = key_warps * 32;
  const int key0w = key_warp * kKeysPerWarp;
  const bool second_key_tile = 2 * key_warp + 1 < layout.key_tiles;
  const int g = lane >> 2, column = (lane & 3) * 2;
  const int first_tile = group * layout.group_tiles, last_tile = min(p.q_tiles, first_tile + layout.group_tiles);
  __nv_bfloat16* group_ds = ds + group * key_warps * 16 * layout.ds_pitch;
  // The warp's 32 keys lie in one 64-key chunk; the forward's keep words of query columns c = 0, 1 (lanes 8 (lane % 4)
  // + lane / 8 + 4 c) hold this lane's bits at 8 m + 4 half + 2 n after the shift.
  const int chunk = key0w / kChunk;
  const int keep_lane = 8 * (lane & 3) + (lane >> 3), keep_shift = 16 * (key_warp & 1) + (g & 1);
  const float scale = p.scale_log2 / kLog2e;
  float dk[2][4][4], dv[2][4][4];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
      for (int i = 0; i < 4; ++i) dk[m][n][i] = dv[m][n][i] = 0.f;
    }
  }

  // The raw key pairs of the key norm backward, loaded during the last round (group 0).
  __nv_bfloat162 k_raw[2][2][4];
  // The warp's main loop over its key tiles (2, or 1 for a last warp of 16 keys).
  auto key_loop = [&](auto key_tiles) {
    constexpr int kKeyTiles = decltype(key_tiles)::value;
    for (int round0 = first_tile; round0 < last_tile; round0 += key_warps) {
      const int round_end = min(last_tile, round0 + key_warps);
      for (int tile = round0; tile < round_end; ++tile) {
        const int query0 = tile * 16;
        __nv_bfloat16* slot = group_ds + (tile - round0) * 16 * layout.ds_pitch;
        if (kMask && !p.tile_attended[tile * p.chunks + chunk]) {
          // Nothing attended: the warp's part of dS is zero.
          for (int index = lane; index < 16 * 4; index += 32) {
            *reinterpret_cast<uint4*>(slot + (index >> 2) * layout.ds_pitch + key0w + (index & 3) * 8) =
                make_uint4(0u, 0u, 0u, 0u);
          }
          continue;
        }
        uint32_t q_b[2][4], grad_b[2][4];
        load_b_rows(q_b[0], q_tile, query0, lane);
        load_b_rows(q_b[1], q_tile, query0 + 8, lane);
        load_b_rows(grad_b[0], grad_tile, query0, lane);
        load_b_rows(grad_b[1], grad_tile, query0 + 8, lane);
        float2 neg_lse[2], delta2[2];
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          const float2 lse = *reinterpret_cast<const float2*>(lse_rows + query0 + 8 * n + column);
          neg_lse[n] = make_float2(-lse.x, -lse.y);
          delta2[n] = *reinterpret_cast<const float2*>(delta_rows + query0 + 8 * n + column);
        }
        uint32_t keep[2] = {0xffffffffu, 0xffffffffu};
        if (kDropout) {
          const uint32_t* words = bits + (tile * p.chunks + chunk) * 32 + keep_lane;
          keep[0] = words[0] >> keep_shift;
          keep[1] = words[4] >> keep_shift;
        }
        // [n][c]: the attend bits of the warp's keys for query 8 n + column + c, this lane's at 16 m + 8 half.
        uint32_t attend[2][2] = {};
        if (kMask) {
#pragma unroll
          for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
              attend[n][c] = mask_words[(query0 + 8 * n + column + c) * 2 * p.chunks + key_warp] >> g;
            }
          }
        }
        // Transposed scores and output-gradient products: rows are keys, columns queries.
        float s[kKeyTiles][2][4], dp[kKeyTiles][2][4];
#pragma unroll
        for (int m = 0; m < kKeyTiles; ++m) {
          uint32_t k_a[2][4], v_a[2][4];
          load_a(k_a[0], k_tile, key0w + 16 * m, 0, lane);
          load_a(k_a[1], k_tile, key0w + 16 * m, 16, lane);
          load_a(v_a[0], v_tile, key0w + 16 * m, 0, lane);
          load_a(v_a[1], v_tile, key0w + 16 * m, 16, lane);
#pragma unroll
          for (int n = 0; n < 2; ++n) {
            s[m][n][0] = s[m][n][1] = s[m][n][2] = s[m][n][3] = 0.f;
            dp[m][n][0] = dp[m][n][1] = dp[m][n][2] = dp[m][n][3] = 0.f;
            mma(s[m][n], k_a[0], q_b[n][0], q_b[n][1]);
            mma(s[m][n], k_a[1], q_b[n][2], q_b[n][3]);
            mma(dp[m][n], v_a[0], grad_b[n][0], grad_b[n][1]);
            mma(dp[m][n], v_a[1], grad_b[n][2], grad_b[n][3]);
          }
        }
        uint32_t a_p[kKeyTiles][4], a_ds[kKeyTiles][4];
#pragma unroll
        for (int m = 0; m < kKeyTiles; ++m) {
          const int key_row0 = key0w + 16 * m;
          const bool tail = !kMask && key_row0 + 16 > p.keys;
#pragma unroll
          for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int half = e >> 1, c = e & 1;
              const float neg_row_lse = c ? neg_lse[n].y : neg_lse[n].x, delta = c ? delta2[n].y : delta2[n].x;
              float probability = fast_exp2(fmaf(s[m][n][e], p.scale_log2, neg_row_lse));
              if (kMask) {
                probability = (attend[n][c] >> (16 * m + 8 * half)) & 1u ? probability : 0.f;
              } else if (tail) {
                probability = key_row0 + g + 8 * half < p.keys ? probability : 0.f;
              }
              const bool kept = !kDropout || ((keep[c] >> (8 * m + 4 * half + 2 * n)) & 1u);
              // The keep scale of the dropped probabilities goes onto the value gradient at the end.
              s[m][n][e] = kept ? probability : 0.f;
              dp[m][n][e] = probability * (kept ? fmaf(dp[m][n][e], p.keep_scale, -delta) : -delta);
            }
          }
          accumulators_to_a(a_p[m], s[m][0], s[m][1]);
          accumulators_to_a(a_ds[m], dp[m][0], dp[m][1]);
        }
        uint32_t q_t[2][4], grad_t[2][4];
        load_b_trans(q_t[0], q_tile, query0, 0, lane);
        load_b_trans(q_t[1], q_tile, query0, 16, lane);
        load_b_trans(grad_t[0], grad_tile, query0, 0, lane);
        load_b_trans(grad_t[1], grad_tile, query0, 16, lane);
#pragma unroll
        for (int m = 0; m < kKeyTiles; ++m) {
#pragma unroll
          for (int n = 0; n < 2; ++n) {
            mma(dv[m][2 * n], a_p[m], grad_t[n][0], grad_t[n][1]);
            mma(dv[m][2 * n + 1], a_p[m], grad_t[n][2], grad_t[n][3]);
            mma(dk[m][2 * n], a_ds[m], q_t[n][0], q_t[n][1]);
            mma(dk[m][2 * n + 1], a_ds[m], q_t[n][2], q_t[n][3]);
          }
          stmatrix_x4_trans(
              slot + ((lane >> 4) * 8 + (lane & 7)) * layout.ds_pitch + key0w + 16 * m + ((lane >> 3) & 1) * 8,
              a_ds[m]);
        }
      }
      group_barrier(group, group_threads);
      if (round0 + key_warps >= last_tile && group == 0) {
#pragma unroll
        for (int m = 0; m < kKeyTiles; ++m) {
          const int key_a = key0w + 16 * m + g;
          load_raw_pairs(k_raw[m][0], kv_rows + min(key_a, p.keys - 1) * p.kv_row_stride + p.k_col + head * kHeadDim,
                         column);
          load_raw_pairs(k_raw[m][1],
                         kv_rows + min(key_a + 8, p.keys - 1) * p.kv_row_stride + p.k_col + head * kHeadDim, column);
        }
      }
      const int dq_tile = round0 + key_warp;
      if (dq_tile < round_end) {
        const __nv_bfloat16* slot = group_ds + key_warp * 16 * layout.ds_pitch;
        const int local_a = dq_tile * 16 + g, local_b = local_a + 8;
        __nv_bfloat162 raw_a[4], raw_b[4];
        load_raw_pairs(raw_a, q_rows + min(local_a, p.queries - 1) * p.q_row_stride + p.q_col + head * kHeadDim,
                       column);
        load_raw_pairs(raw_b, q_rows + min(local_b, p.queries - 1) * p.q_row_stride + p.q_col + head * kHeadDim,
                       column);
        float dq[4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n) dq[n][0] = dq[n][1] = dq[n][2] = dq[n][3] = 0.f;
        for (int kk = 0; kk < layout.key_tiles; ++kk) {
          uint32_t a[4], b0[4], b1[4];
          load_a(a, slot, 0, 16 * kk, lane, layout.ds_pitch);
          load_b_trans(b0, k_tile, 16 * kk, 0, lane);
          load_b_trans(b1, k_tile, 16 * kk, 16, lane);
          mma(dq[0], a, b0[0], b0[1]);
          mma(dq[1], a, b0[2], b0[3]);
          mma(dq[2], a, b1[0], b1[1]);
          mma(dq[3], a, b1[2], b1[3]);
        }
        // The normalized queries' gradient in bf16, as autocast hands it to the norm's backward.
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const bool valid = (i < 2 ? local_a : local_b) < p.queries;
            dq[n][i] = valid ? __bfloat162float(__float2bfloat16_rn(dq[n][i] * scale)) : 0.f;
          }
        }
        float dq_raw[4][4];
        float q_weight_grad[8] = {}, q_bias_grad[8] = {};
        norm_backward(dq, raw_a, raw_b, q_stats[local_a], q_stats[local_b], p.q_weight, column, dq_raw, q_weight_grad,
                      q_bias_grad);
        store_rows(p.dq_src + batch * p.dq_batch_stride + p.q_col + head * kHeadDim, p.dq_row_stride, local_a, local_b,
                   p.queries, column, dq_raw);
        add_norm_partials(q_weight_grad, q_bias_grad, q_sums + warp * 64, lane);
      }
      group_barrier(group, group_threads);
    }
  };
  if (second_key_tile) {
    key_loop(std::integral_constant<int, 2>{});
  } else {
    key_loop(std::integral_constant<int, 1>{});
  }

  if (layout.groups > 1) {
    __syncthreads();
    float* partials = reinterpret_cast<float*>(shared_raw + layout.q_tile);
    if (group > 0) {
      float* mine = partials + ((group - 1) * key_warps + key_warp) * kKeyGradientFloats;
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            mine[((m * 4 + n) * 4 + i) * 32 + lane] = dk[m][n][i];
            mine[kKeyGradientFloats / 2 + ((m * 4 + n) * 4 + i) * 32 + lane] = dv[m][n][i];
          }
        }
      }
    }
    __syncthreads();
    if (group == 0) {
      for (int other = 1; other < layout.groups; ++other) {
        const float* theirs = partials + ((other - 1) * key_warps + key_warp) * kKeyGradientFloats;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
          for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              dk[m][n][i] += theirs[((m * 4 + n) * 4 + i) * 32 + lane];
              dv[m][n][i] += theirs[kKeyGradientFloats / 2 + ((m * 4 + n) * 4 + i) * 32 + lane];
            }
          }
        }
      }
    }
  }
  float k_weight_grad[8] = {}, k_bias_grad[8] = {};
  if (group == 0) {
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      if (m == 1 && !second_key_tile) break;
      const int key_a = key0w + 16 * m + g, key_b = key_a + 8;
#pragma unroll
      for (int n = 0; n < 4; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const bool valid = (i < 2 ? key_a : key_b) < p.keys;
          dk[m][n][i] = valid ? __bfloat162float(__float2bfloat16_rn(dk[m][n][i] * scale)) : 0.f;
        }
      }
      float dk_raw[4][4];
      norm_backward(dk[m], k_raw[m][0], k_raw[m][1], k_stats[key_a], k_stats[key_b], p.k_weight, column, dk_raw,
                    k_weight_grad, k_bias_grad);
      __nv_bfloat16* dkv_rows = p.dkv_src + batch * p.dkv_batch_stride + head * kHeadDim;
      store_rows(dkv_rows + p.k_col, p.dkv_row_stride, key_a, key_b, p.keys, column, dk_raw);
#pragma unroll
      for (int n = 0; n < 4; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) dv[m][n][i] *= p.keep_scale;
      }
      store_rows(dkv_rows + p.v_col, p.dkv_row_stride, key_a, key_b, p.keys, column, dv[m]);
    }
  }
  __syncthreads();
  for (int index = threadIdx.x; index < 64; index += threads) {
    float total = 0.f;
    for (int w = 0; w < (threads >> 5); ++w) total += q_sums[w * 64 + index];
    p.q_norm_partial[batch_head * 64 + index] = total;
  }
  reduce_norm_partials(k_weight_grad, k_bias_grad, k_scratch, p.k_norm_partial + batch_head * 64);
}

// Dense keep decisions [batch, heads, queries, keys] of a forward's keep bits, for tests; chunks a query tile skipped
// (nothing attended) read as kept.
__global__ void dropout_keep_kernel(const uint32_t* keep_bits, bool* keep, int batch_heads, int queries, int keys,
                                    int q_tiles, int chunks, const bool* mask) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(batch_heads) * queries * keys;
  if (index >= total) return;
  const int key_index = static_cast<int>(index % keys);
  const int query = static_cast<int>((index / keys) % queries);
  const int batch_head = static_cast<int>(index / (static_cast<int64_t>(keys) * queries));
  const int tile = query / 16, local_query = query % 16, chunk = key_index / kChunk, key_in_chunk = key_index % kChunk;
  const int source_lane = (local_query & 7) * 4 + ((key_in_chunk & 7) >> 1);
  const int bit = (key_in_chunk >> 3) * 4 + (local_query >= 8 ? 2 : 0) + (key_in_chunk & 1);
  const bool masked = mask != nullptr && !mask[static_cast<int64_t>(query) * keys + key_index];
  keep[index] =
      masked ||
      ((keep_bits[((static_cast<int64_t>(batch_head) * q_tiles + tile) * chunks + chunk) * 32 + source_lane] >> bit) &
       1u);
}

void check_source(const at::Tensor& source, const char* name) {
  TORCH_CHECK(source.is_cuda() && source.scalar_type() == at::kBFloat16, name, " must be a CUDA bf16 tensor");
  TORCH_CHECK(source.dim() == 3 && source.stride(2) == 1, name,
              " must be [batch, rows, columns] with unit column stride");
  TORCH_CHECK(source.stride(1) % 8 == 0 && source.stride(0) % 8 == 0, name, " rows must be 16-byte aligned");
}

void check_norm(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.is_cuda() && tensor.scalar_type() == at::kFloat && tensor.numel() == kHeadDim && tensor.is_contiguous(),
      name, " must be a contiguous CUDA fp32 tensor of ", kHeadDim);
}

// Checks the inputs and fills the Params both directions share; the caller sets the outputs and the mask tables.
Params make_params(const at::Tensor& q_src, const at::Tensor& kv_src, int64_t q_col, int64_t k_col, int64_t v_col,
                   int64_t heads, const at::Tensor& q_weight, const at::Tensor& q_bias, const at::Tensor& k_weight,
                   const at::Tensor& k_bias, const std::optional<at::Tensor>& mask, double dropout_p, double eps) {
  check_source(q_src, "q_src");
  check_source(kv_src, "kv_src");
  check_norm(q_weight, "q_weight");
  check_norm(q_bias, "q_bias");
  check_norm(k_weight, "k_weight");
  check_norm(k_bias, "k_bias");
  TORCH_CHECK(q_src.size(0) == kv_src.size(0), "q_src and kv_src must have the same batch");
  TORCH_CHECK(q_col % 8 == 0 && k_col % 8 == 0 && v_col % 8 == 0, "column offsets must be multiples of 8");
  TORCH_CHECK(q_col + heads * kHeadDim <= q_src.size(2), "q columns out of range");
  TORCH_CHECK(k_col + heads * kHeadDim <= kv_src.size(2) && v_col + heads * kHeadDim <= kv_src.size(2),
              "k or v columns out of range");
  Params p{};
  p.q_src = reinterpret_cast<const __nv_bfloat16*>(q_src.data_ptr());
  p.kv_src = reinterpret_cast<const __nv_bfloat16*>(kv_src.data_ptr());
  p.q_batch_stride = q_src.stride(0);
  p.q_row_stride = q_src.stride(1);
  p.kv_batch_stride = kv_src.stride(0);
  p.kv_row_stride = kv_src.stride(1);
  p.q_col = static_cast<int>(q_col);
  p.k_col = static_cast<int>(k_col);
  p.v_col = static_cast<int>(v_col);
  p.batch = static_cast<int>(q_src.size(0));
  p.heads = static_cast<int>(heads);
  p.queries = static_cast<int>(q_src.size(1));
  p.keys = static_cast<int>(kv_src.size(1));
  TORCH_CHECK(p.queries > 0 && p.keys > 0, "empty attention");
  TORCH_CHECK(p.keys <= kMaxKeys, "planner_attention holds at most ", kMaxKeys, " keys, got ", p.keys);
  p.q_tiles = (p.queries + 15) / 16;
  p.chunks = (p.keys + kChunk - 1) / kChunk;
  p.q_weight = q_weight.data_ptr<float>();
  p.q_bias = q_bias.data_ptr<float>();
  p.k_weight = k_weight.data_ptr<float>();
  p.k_bias = k_bias.data_ptr<float>();
  if (mask.has_value()) {
    TORCH_CHECK(mask->is_cuda() && mask->scalar_type() == at::kBool && mask->is_contiguous() && mask->dim() == 2 &&
                    mask->size(0) == p.queries && mask->size(1) == p.keys,
                "mask must be a contiguous CUDA bool [queries, keys]");
    p.mask = mask->data_ptr<bool>();
  }
  p.eps = static_cast<float>(eps);
  p.scale_log2 = kLog2e / sqrtf(static_cast<float>(kHeadDim));
  p.drop_threshold = drop_threshold(dropout_p);
  p.keep_scale = static_cast<float>(1.0 / (1.0 - dropout_p));
  return p;
}

// Lets kernel use more than 48 KB of dynamic shared memory where it needs it.
void set_shared(const void* kernel, size_t bytes) {
  if (bytes > 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
  }
}

// With a mask, computes its bits and per-tile attended table (see Params) and points p at them.
std::tuple<at::Tensor, at::Tensor> mask_tables(Params& p, const at::Tensor& like) {
  if (p.mask == nullptr) return {};
  auto words = at::empty({static_cast<int64_t>(p.q_tiles) * 16 * 2 * p.chunks}, like.options().dtype(at::kInt));
  auto attended = at::empty({static_cast<int64_t>(p.q_tiles) * p.chunks}, like.options().dtype(at::kByte));
  mask_words_kernel<<<p.q_tiles, 16 * 2 * p.chunks, 0, at::cuda::getCurrentCUDAStream()>>>(
      p.mask, p.queries, p.keys, p.chunks, reinterpret_cast<uint32_t*>(words.data_ptr()), attended.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  p.mask_words = reinterpret_cast<const uint32_t*>(words.data_ptr());
  p.tile_attended = attended.data_ptr<uint8_t>();
  return {words, attended};
}

// The kernel instance for a mask (or none) and dropout (or none).
template <template <bool, bool> class Kernel>
auto select_kernel(bool masked, bool dropout) {
  if (masked) return dropout ? Kernel<true, true>::value : Kernel<true, false>::value;
  return dropout ? Kernel<false, true>::value : Kernel<false, false>::value;
}

template <bool kMask, bool kDropout>
struct ForwardKernel {
  static constexpr auto value = attention_forward_kernel<kMask, kDropout>;
};

template <bool kMask, bool kDropout>
struct BackwardKernel {
  static constexpr auto value = attention_backward_kernel<kMask, kDropout>;
};

// Forward: out [batch, queries, heads * 32] bf16, lse [batch, heads, queries] fp32 in the log2 domain, and the keep
// bits (empty without dropout).
std::tuple<at::Tensor, at::Tensor, at::Tensor> planner_attention(
    const at::Tensor& q_src, const at::Tensor& kv_src, int64_t q_col, int64_t k_col, int64_t v_col, int64_t heads,
    const at::Tensor& q_weight, const at::Tensor& q_bias, const at::Tensor& k_weight, const at::Tensor& k_bias,
    const std::optional<at::Tensor>& mask, double dropout_p, double eps, bool shared) {
  const c10::cuda::CUDAGuard guard(q_src.device());
  TORCH_CHECK(!shared || q_src.data_ptr() == kv_src.data_ptr(), "shared needs one source for q, k and v");
  Params p =
      make_params(q_src, kv_src, q_col, k_col, v_col, heads, q_weight, q_bias, k_weight, k_bias, mask, dropout_p, eps);
  const bool dropout = p.drop_threshold != 0u;
  auto out = at::empty({p.batch, p.queries, heads * kHeadDim}, q_src.options());
  auto lse = at::empty({p.batch, heads, p.queries}, q_src.options().dtype(at::kFloat));
  const int64_t words = dropout ? static_cast<int64_t>(p.batch) * heads * p.q_tiles * p.chunks * 32 : 0;
  auto keep_bits = at::empty({words}, q_src.options().dtype(at::kInt));
  p.philox = dropout ? philox_state() : at::PhiloxCudaState(0, 0);
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.lse = lse.data_ptr<float>();
  p.keep_bits = reinterpret_cast<uint32_t*>(keep_bits.data_ptr());
  const auto tables = mask_tables(p, q_src);
  const int warps = kForwardThreads / 32;
  p.forward_tiles = 3 * warps;
  const dim3 grid(p.batch * p.heads, (p.q_tiles + p.forward_tiles - 1) / p.forward_tiles);
  const size_t shared_bytes = (2 * round_up16(p.keys) + warps * 16) * kPitch * sizeof(__nv_bfloat16);
  const auto kernel = select_kernel<ForwardKernel>(p.mask != nullptr, dropout);
  set_shared(reinterpret_cast<const void*>(kernel), shared_bytes);
  kernel<<<grid, kForwardThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse, keep_bits};
}

// Backward: the gradients of q_src and kv_src (kv_src's empty when shared, as q_src's holds every column) and the [32]
// weight and bias gradients of the query and key norms.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> planner_attention_backward(
    const at::Tensor& grad, const at::Tensor& q_src, const at::Tensor& kv_src, int64_t q_col, int64_t k_col,
    int64_t v_col, int64_t heads, const at::Tensor& q_weight, const at::Tensor& q_bias, const at::Tensor& k_weight,
    const at::Tensor& k_bias, const std::optional<at::Tensor>& mask, double dropout_p, const at::Tensor& keep_bits,
    double eps, const at::Tensor& out, const at::Tensor& lse, bool shared) {
  const c10::cuda::CUDAGuard guard(q_src.device());
  Params p =
      make_params(q_src, kv_src, q_col, k_col, v_col, heads, q_weight, q_bias, k_weight, k_bias, mask, dropout_p, eps);
  const bool dropout = p.drop_threshold != 0u;
  const auto grad_contiguous = grad.contiguous();
  TORCH_CHECK(out.is_contiguous() && lse.is_contiguous(), "out and lse must be contiguous");
  TORCH_CHECK(grad_contiguous.sizes() == out.sizes(), "grad must have the output's shape");
  TORCH_CHECK(!dropout || keep_bits.numel() == static_cast<int64_t>(p.batch) * heads * p.q_tiles * p.chunks * 32,
              "keep_bits does not match the forward: ", keep_bits.numel(), " words for dropout ", dropout_p);
  // Self-attention's one source gets one gradient with every column written; separate sources each get their own.
  at::Tensor dq_src, dkv_src;
  if (shared) {
    TORCH_CHECK(q_src.size(2) == 3 * heads * kHeadDim, "a shared source must hold exactly q, k and v");
    dq_src = at::empty(q_src.sizes(), q_src.options());
    dkv_src = dq_src;
  } else {
    TORCH_CHECK(q_src.size(2) == heads * kHeadDim && kv_src.size(2) == 2 * heads * kHeadDim,
                "separate sources must hold exactly q, and k and v");
    dq_src = at::empty(q_src.sizes(), q_src.options());
    dkv_src = at::empty(kv_src.sizes(), kv_src.options());
  }
  auto q_partial = at::empty({static_cast<int64_t>(p.batch) * heads, 64}, lse.options());
  auto k_partial = at::empty({static_cast<int64_t>(p.batch) * heads, 64}, lse.options());
  p.keep_bits = reinterpret_cast<uint32_t*>(const_cast<void*>(keep_bits.data_ptr()));
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.lse = lse.data_ptr<float>();
  p.grad = reinterpret_cast<const __nv_bfloat16*>(grad_contiguous.data_ptr());
  p.dq_src = reinterpret_cast<__nv_bfloat16*>(dq_src.data_ptr());
  p.dkv_src = reinterpret_cast<__nv_bfloat16*>(dkv_src.data_ptr());
  p.dq_batch_stride = dq_src.stride(0);
  p.dq_row_stride = dq_src.stride(1);
  p.dkv_batch_stride = dkv_src.stride(0);
  p.dkv_row_stride = dkv_src.stride(1);
  p.q_norm_partial = q_partial.data_ptr<float>();
  p.k_norm_partial = k_partial.data_ptr<float>();
  const auto tables = mask_tables(p, q_src);

  const BackwardLayout layout = backward_layout(p.keys, p.q_tiles, p.chunks, dropout, p.mask != nullptr);
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(layout.total <= properties->sharedMemPerBlockOptin, "planner_attention_backward needs ", layout.total,
              " bytes of shared memory per block for ", p.queries, " queries and ", p.keys, " keys; ", properties->name,
              " allows ", properties->sharedMemPerBlockOptin);
  TORCH_CHECK(layout.groups == 1 ||
                  static_cast<size_t>(layout.groups - 1) * layout.key_warps * kKeyGradientFloats * sizeof(float) <=
                      layout.k_stats - layout.q_tile,
              "planner_attention_backward: the query tiles cannot hold the groups' key gradients");
  const auto kernel = select_kernel<BackwardKernel>(p.mask != nullptr, dropout);
  set_shared(reinterpret_cast<const void*>(kernel), layout.total);
  kernel<<<p.batch * p.heads, layout.key_warps * layout.groups * 32, layout.total, at::cuda::getCurrentCUDAStream()>>>(
      p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const auto q_norm_grad = column_sum(q_partial);
  const auto k_norm_grad = column_sum(k_partial);
  if (shared) dkv_src = at::empty({0}, kv_src.options());
  return {dq_src,
          dkv_src,
          q_norm_grad.slice(0, 0, kHeadDim).clone(),
          q_norm_grad.slice(0, kHeadDim, 2 * kHeadDim).clone(),
          k_norm_grad.slice(0, 0, kHeadDim).clone(),
          k_norm_grad.slice(0, kHeadDim, 2 * kHeadDim).clone()};
}

int64_t planner_attention_backward_shared_bytes(int64_t queries, int64_t keys, bool dropout, bool mask) {
  const int q_tiles = static_cast<int>((queries + 15) / 16), chunks = static_cast<int>((keys + kChunk - 1) / kChunk);
  return static_cast<int64_t>(backward_layout(static_cast<int>(keys), q_tiles, chunks, dropout, mask).total);
}

// The dense bool keep decisions [batch, heads, queries, keys] of a forward's keep bits, for tests.
at::Tensor planner_dropout_keep(const at::Tensor& keep_bits, int64_t batch, int64_t heads, int64_t queries,
                                int64_t keys, const std::optional<at::Tensor>& mask) {
  const c10::cuda::CUDAGuard guard(keep_bits.device());
  const int q_tiles = static_cast<int>((queries + 15) / 16), chunks = static_cast<int>((keys + kChunk - 1) / kChunk);
  TORCH_CHECK(keep_bits.numel() == batch * heads * q_tiles * chunks * 32, "keep_bits does not match the shape");
  auto keep = at::empty({batch, heads, queries, keys}, keep_bits.options().dtype(at::kBool));
  const int64_t total = keep.numel();
  if (total == 0) return keep;
  dropout_keep_kernel<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint32_t*>(keep_bits.data_ptr()), keep.data_ptr<bool>(), static_cast<int>(batch * heads),
      static_cast<int>(queries), static_cast<int>(keys), q_tiles, chunks,
      mask.has_value() ? mask->data_ptr<bool>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return keep;
}

}  // namespace

TORCH_LIBRARY_IMPL(py123d_garage_cuda, CUDA, m) {
  m.impl("planner_attention", &planner_attention);
  m.impl("planner_attention_backward", &planner_attention_backward);
  m.impl("planner_dropout_keep", &planner_dropout_keep);
}

TORCH_LIBRARY_IMPL(py123d_garage_cuda, CompositeExplicitAutograd, m) {
  m.impl("planner_attention_backward_shared_bytes", &planner_attention_backward_shared_bytes);
}

}  // namespace py123d_garage_cuda
