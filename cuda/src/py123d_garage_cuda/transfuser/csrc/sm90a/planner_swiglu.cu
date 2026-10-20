// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// The planner feed-forward's first projection and activation, h = dropout(silu(gate) * value) of [gate | value] =
// x weight^T + bias, as one Hopper GEMM (TMA + wgmma) with the activation and dropout in its epilogue; [gate | value]
// leaves for the backward by an asynchronous copy, so the activation reads it from shared memory instead of global
// memory. [gate | value] rounds to bf16 like the compiled torch reference; the backward redraws the dropout mask.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include <cstdint>

#include "transfuser/csrc/arch.cuh"
#include "transfuser/csrc/sm80/planner.cuh"
#include "transfuser/csrc/sm90/tma.cuh"

namespace py123d_garage_cuda {
namespace {

using namespace planner;

// Tiles of 128 rows x 64 hidden columns, so every tile row stores whole 128-byte lines; the GEMM's N holds the tile's
// 64 gate and 64 value columns, its K (at most 256) runs in 64-wide blocks.
constexpr int kTileRows = 128;
constexpr int kTileWidth = 64;
constexpr int kTileN = 2 * kTileWidth;
constexpr int kBlockK = 64;
constexpr int kMaxKBlocks = 4;
// Warps 0-7: two MMA warpgroups of 64 rows each; warps 8-14: the epilogue; warp 15: the loads.
constexpr int kMmaWarps = 8;
// Registers per thread of the MMA warpgroups and of the others, set at the start of their roles.
[[maybe_unused]] constexpr int kMmaRegisters = 152;
[[maybe_unused]] constexpr int kOtherRegisters = 104;
constexpr int kEpilogueWarps = 7;
constexpr int kThreads = (kMmaWarps + kEpilogueWarps + 1) * 32;
constexpr int kAccumulators = kTileN / 2;
// The epilogue's items are (row, 8-column group) pairs: thread s * 8 + k takes group k of rows s + 28 i.
constexpr int kGroups = kTileWidth / 8;
constexpr int kItemThreads = kEpilogueWarps * 32;
constexpr int kRowStride = kItemThreads / kGroups;
constexpr int kMaxItems = (kTileRows + kRowStride - 1) / kRowStride;
constexpr int kStages = 4;
constexpr int kABytes = kTileRows * kBlockK * 2;
constexpr int kHalfBBytes = kTileWidth * kBlockK * 2;
constexpr int kBBlockBytes = 2 * kHalfBBytes;
constexpr int kOutBytes = kTileRows * kTileWidth * 2;
// Two buffers of a tile's bf16 gate tile and value tile, from the MMA warpgroups to the epilogue; their 128-byte rows
// hold the 16-byte groups swizzled like the TMA's 128-byte mode (group k of row r at k ^ (r % 8)).
constexpr int kBuffers = 2;
constexpr int kBufferBytes = 2 * kOutBytes;
// Shared memory: the block's current weight tile (gate and value rows per K block), the x stages, the buffers and the
// barriers (full and empty per stage and per buffer, and per weight tile). A buffer also leaves as [gate | value].
constexpr int kStageOffset = kMaxKBlocks * kBBlockBytes;
constexpr int kBufferOffset = kStageOffset + kStages * kABytes;
constexpr int kBarrierOffset = kBufferOffset + kBuffers * kBufferBytes;
constexpr int kBarriers = 2 * kStages + 2 * kBuffers + 2;
constexpr int kSharedBytes = kBarrierOffset + 8 * kBarriers + 1024;

struct GemmParams {
  int rows, width, k_blocks, row_tiles, tiles;
  const __nv_bfloat16* bias;  // [2 width]
  uint32_t drop_threshold;    // 0 keeps everything
  float keep_scale;
  at::PhiloxCudaState philox;  // forward
  int64_t* rng;                // written with the stream's (seed, offset)
  __nv_bfloat16* out;          // the activation [rows, width]
};

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void mbarrier_init(uint32_t barrier, int count) {
  PY123D_SM90A(asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(barrier), "r"(count)));
}

__device__ __forceinline__ void mbarrier_expect(uint32_t barrier, int bytes) {
  PY123D_SM90A(
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(barrier), "r"(bytes) : "memory"));
}

__device__ __forceinline__ void mbarrier_arrive(uint32_t barrier) {
  PY123D_SM90A(asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(barrier) : "memory"));
}

__device__ __forceinline__ void mbarrier_wait(uint32_t barrier, uint32_t parity) {
  PY123D_SM90A(
      asm volatile("{\n"
                   ".reg .pred done;\n"
                   "WAIT_%=:\n"
                   "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
                   "@!done bra WAIT_%=;\n"
                   "}\n" ::"r"(barrier),
                   "r"(parity) : "memory"));
}

__device__ __forceinline__ void tma_load_2d(uint32_t destination, const CUtensorMap* map, int c0, int c1,
                                            uint32_t barrier) {
  PY123D_SM90A(
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, "
                   "%3}], [%4];\n" ::"r"(destination),
                   "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(barrier) : "memory"));
}

__device__ __forceinline__ void tma_load_3d(uint32_t destination, const CUtensorMap* map, int c0, int c1, int c2,
                                            uint32_t barrier) {
  PY123D_SM90A(asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], "
      "[%5];\n" ::"r"(destination),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(c2), "r"(barrier) : "memory"));
}

// The wgmma descriptor of a K-major operand in 128-byte swizzled rows of 64 bf16, 8-row groups 1024 bytes apart.
__device__ __forceinline__ uint64_t matrix_descriptor(uint32_t address) {
  return static_cast<uint64_t>((address & 0x3ffffu) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}

__device__ __forceinline__ void wgmma_fence(float (&d)[kAccumulators]) {
#pragma unroll
  for (int i = 0; i < kAccumulators; ++i) asm volatile("" : "+f"(d[i])::"memory");
  PY123D_SM90A(asm volatile("wgmma.fence.sync.aligned;\n" :: : "memory"));
}

__device__ __forceinline__ void wgmma_commit() {
  PY123D_SM90A(asm volatile("wgmma.commit_group.sync.aligned;\n" :: : "memory"));
}

template <int N>
__device__ __forceinline__ void wgmma_wait(float (&d)[kAccumulators]) {
  PY123D_SM90A(asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"));
#pragma unroll
  for (int i = 0; i < kAccumulators; ++i) asm volatile("" : "+f"(d[i])::"memory");
}

// d (+)= a b^T for a 64 x 16 tile of a and a 128 x 16 tile of b, both K-major in shared memory; accumulate false
// overwrites d.
__device__ __forceinline__ void wgmma_m64n128(float (&d)[kAccumulators], uint64_t a, uint64_t b, bool accumulate) {
  PY123D_SM90A(asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, "
      "%46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, "
      "0, "
      "0;\n"
      "}\n" : "+f"(d[0]),
      "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
      "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
      "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
      "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]),
      "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
      "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]),
      "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]),
      "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "l"(a), "l"(b),
      "r"(static_cast<uint32_t>(accumulate))));
}

__device__ __forceinline__ float sigmoid(float x) {
  float power, reciprocal;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(power) : "f"(x * -1.4426950408889634f));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(1.f + power));
  return reciprocal;
}

__device__ __forceinline__ void mma_barrier() { asm volatile("bar.sync 1, %0;\n" ::"n"(kMmaWarps * 32) : "memory"); }

__device__ __forceinline__ void tma_store_3d(const CUtensorMap* map, int c0, int c1, int c2, uint32_t source) {
  PY123D_SM90A(asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];\n" ::"l"(
                                reinterpret_cast<uint64_t>(map)),
                            "r"(c0), "r"(c1), "r"(c2), "r"(source) : "memory"));
}

__device__ __forceinline__ void tma_store_commit() {
  PY123D_SM90A(asm volatile("cp.async.bulk.commit_group;\n" :: : "memory"));
}

// Waits until at most N committed stores still read shared memory.
template <int N>
__device__ __forceinline__ void tma_store_wait_read() {
  PY123D_SM90A(asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory"));
}

// Makes this thread's shared-memory writes visible to the TMA.
__device__ __forceinline__ void fence_async_shared() {
  PY123D_SM90A(asm volatile("fence.proxy.async.shared::cta;\n" :: : "memory"));
}

__device__ __forceinline__ uint32_t pack_bias(float a, float b, __nv_bfloat162 bias) {
  const float2 pair = __bfloat1622float2(bias);
  const __nv_bfloat162 value = __floats2bfloat162_rn(a + pair.x, b + pair.y);
  return *reinterpret_cast<const uint32_t*>(&value);
}

// Four 8 x 8 bf16 matrices from registers into shared memory; lane l gives row l % 8 of matrix l / 8.
__device__ __forceinline__ void stmatrix_x4(uint32_t address, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  PY123D_SM90A(asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(address),
                            "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory"));
}

// The byte offset of 16-byte group `group` of row `row` in a swizzled buffer tile.
__device__ __forceinline__ uint32_t swizzled(int row, int group) {
  return static_cast<uint32_t>(row * kTileWidth * 2 + ((group ^ (row % 8)) * 16));
}

// The warp's 16 rows of one half (gate: half 0, value: half 1) of the accumulators, plus bias, as bf16 into a swizzled
// tile at rows [row0, +16). Accumulator 4 j + 2 r + c is row lane / 4 + 8 r, column 8 j + 2 (lane % 4) + c of the
// gate half, and 4 (j + kGroups) + 2 r + c the same value column.
__device__ __forceinline__ void store_half(const float (&acc)[kAccumulators], int half,
                                           const __nv_bfloat162 (&bias)[kGroups], uint32_t tile, int row0, int lane) {
  const int matrix = lane / 8, row = row0 + lane % 8 + 8 * (matrix % 2);
#pragma unroll
  for (int j = 0; j < kGroups; j += 2) {
    const float* a = acc + 4 * (half * kGroups + j);
    stmatrix_x4(tile + swizzled(row, j + matrix / 2), pack_bias(a[0], a[1], bias[j]), pack_bias(a[2], a[3], bias[j]),
                pack_bias(a[4], a[5], bias[j + 1]), pack_bias(a[6], a[7], bias[j + 1]));
  }
}

// Blocks take contiguous ranges of the [column tiles x row tiles] tiles, row tiles fastest, so a block keeps one weight
// tile in shared memory over many row tiles. The load warp streams the x K blocks; the MMA warpgroups multiply, write
// bf16(acc + bias) to a shared buffer that MMA thread 0 copies out as [gate | value], and hand it to the epilogue
// warps, so a tile's epilogue runs while the next multiplies. Dropout draws 8 16-bit numbers per (row, 8-column group),
// one per column.
__global__ void __launch_bounds__(kThreads, 1)
    swiglu_gemm_kernel(const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap w_map,
                       const __grid_constant__ CUtensorMap gate_value_map, const GemmParams p) {
  extern __shared__ unsigned char shared_raw[];
  const uint32_t raw_address = shared_address(shared_raw);
  const uint32_t base = (raw_address + 1023u) & ~1023u;
  unsigned char* shared = shared_raw + (base - raw_address);
  const uint32_t full = base + kBarrierOffset, empty = full + 8 * kStages;
  const uint32_t buffer_full = empty + 8 * kStages, buffer_empty = buffer_full + 8 * kBuffers;
  const uint32_t weights_full = buffer_empty + 8 * kBuffers, weights_empty = weights_full + 8;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    for (int stage = 0; stage < kStages; ++stage) {
      mbarrier_init(full + 8 * stage, 1);
      mbarrier_init(empty + 8 * stage, kMmaWarps);
    }
    for (int buffer = 0; buffer < kBuffers; ++buffer) {
      mbarrier_init(buffer_full + 8 * buffer, kMmaWarps);
      mbarrier_init(buffer_empty + 8 * buffer, kEpilogueWarps);
    }
    mbarrier_init(weights_full, 1);
    mbarrier_init(weights_empty, kMmaWarps);
    PY123D_SM90A(asm volatile("fence.mbarrier_init.release.cluster;\n" :: : "memory"));
  }
  __syncthreads();
  const int first = static_cast<int>(static_cast<int64_t>(blockIdx.x) * p.tiles / gridDim.x);
  const int last = static_cast<int>(static_cast<int64_t>(blockIdx.x + 1) * p.tiles / gridDim.x);
  if (warp >= kMmaWarps) PY123D_SM90A(asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(kOtherRegisters)));
  if (warp == kMmaWarps + kEpilogueWarps) {
    if (lane != 0) return;
    int stage = 0, column_tile = -1, columns = 0;
    uint32_t phase = 0;
    for (int tile = first; tile < last; ++tile) {
      const int tile_column = tile / p.row_tiles, row0 = tile % p.row_tiles * kTileRows;
      if (tile_column != column_tile) {
        // The MMA warpgroups are done with the previous weight tile.
        if (column_tile >= 0) mbarrier_wait(weights_empty, (columns - 1) & 1);
        column_tile = tile_column;
        ++columns;
        mbarrier_expect(weights_full, p.k_blocks * kBBlockBytes);
        for (int k_block = 0; k_block < p.k_blocks; ++k_block) {
          const uint32_t destination = base + k_block * kBBlockBytes;
          tma_load_3d(destination, &w_map, k_block * kBlockK, column_tile * kTileWidth, 0, weights_full);
          tma_load_3d(destination + kHalfBBytes, &w_map, k_block * kBlockK, column_tile * kTileWidth, 1, weights_full);
        }
      }
      for (int k_block = 0; k_block < p.k_blocks; ++k_block) {
        mbarrier_wait(empty + 8 * stage, phase ^ 1);
        mbarrier_expect(full + 8 * stage, kABytes);
        tma_load_2d(base + kStageOffset + stage * kABytes, &x_map, k_block * kBlockK, row0, full + 8 * stage);
        if (++stage == kStages) {
          stage = 0;
          phase ^= 1;
        }
      }
    }
    return;
  }

  const int64_t row_groups = p.width / 8;
  if (warp < kMmaWarps) {
    PY123D_SM90A(asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(kMmaRegisters)));
    const int group = warp / 4, quad = lane % 4, warp_row0 = group * 64 + warp % 4 * 16;
    float acc[kAccumulators];
    __nv_bfloat162 gate_bias[kGroups], value_bias[kGroups];
    int stage = 0, column_tile = -1, columns = 0;
    uint32_t phase = 0;
    for (int tile = first; tile < last; ++tile) {
      const int tile_column = tile / p.row_tiles, col0 = tile_column * kTileWidth;
      if (tile_column != column_tile) {
        column_tile = tile_column;
#pragma unroll
        for (int j = 0; j < kGroups; ++j) {
          const int col = col0 + 8 * j + 2 * quad;
          gate_bias[j] = value_bias[j] = __floats2bfloat162_rn(0.f, 0.f);
          if (col < p.width) {
            gate_bias[j] = *reinterpret_cast<const __nv_bfloat162*>(p.bias + col);
            value_bias[j] = *reinterpret_cast<const __nv_bfloat162*>(p.bias + p.width + col);
          }
        }
        mbarrier_wait(weights_full, columns & 1);
        ++columns;
      }
      int previous = 0;
      for (int k_block = 0; k_block < p.k_blocks; ++k_block) {
        mbarrier_wait(full + 8 * stage, phase);
        wgmma_fence(acc);
        const uint32_t a = base + kStageOffset + stage * kABytes + group * 64 * 128, b = base + k_block * kBBlockBytes;
#pragma unroll
        for (int k = 0; k < kBlockK / 16; ++k) {
          wgmma_m64n128(acc, matrix_descriptor(a + 32 * k), matrix_descriptor(b + 32 * k), k_block > 0 || k > 0);
        }
        wgmma_commit();
        if (k_block > 0) {
          wgmma_wait<1>(acc);
          if (lane == 0) mbarrier_arrive(empty + 8 * previous);
        }
        previous = stage;
        if (++stage == kStages) {
          stage = 0;
          phase ^= 1;
        }
      }
      wgmma_wait<0>(acc);
      if (lane == 0) {
        mbarrier_arrive(empty + 8 * previous);
        if (tile + 1 < last && (tile + 1) / p.row_tiles != tile_column) mbarrier_arrive(weights_empty);
      }
      const int count = tile - first, buffer = count % kBuffers;
      // The buffer is free once the epilogue has read it and its copy out (the one before the last) has too.
      mbarrier_wait(buffer_empty + 8 * buffer, ((count / kBuffers) & 1) ^ 1);
      if (threadIdx.x == 0) tma_store_wait_read<kBuffers - 1>();
      mma_barrier();
      const uint32_t gate_tile = base + kBufferOffset + buffer * kBufferBytes;
      store_half(acc, 0, gate_bias, gate_tile, warp_row0, lane);
      store_half(acc, 1, value_bias, gate_tile + kOutBytes, warp_row0, lane);
      fence_async_shared();
      mma_barrier();
      if (threadIdx.x == 0) {
        const int row0 = tile % p.row_tiles * kTileRows;
        tma_store_3d(&gate_value_map, col0, 0, row0, gate_tile);
        tma_store_3d(&gate_value_map, col0, 1, row0, gate_tile + kOutBytes);
        tma_store_commit();
      }
      if (lane == 0) mbarrier_arrive(buffer_full + 8 * buffer);
    }
    if (threadIdx.x == 0) tma_store_wait_read<0>();
    return;
  }

  const int thread = threadIdx.x - kMmaWarps * 32, group = thread % kGroups, first_row = thread / kGroups;
  const bool active = thread < kItemThreads;
  const PhiloxStream stream = philox_stream(p.philox, blockIdx.x == 0 && thread == 0 ? p.rng : nullptr);
  for (int tile = first; tile < last; ++tile) {
    const int count = tile - first, buffer = count % kBuffers;
    const int row0 = tile % p.row_tiles * kTileRows, col = tile / p.row_tiles * kTileWidth + 8 * group;
    mbarrier_wait(buffer_full + 8 * buffer, (count / kBuffers) & 1);
    const __nv_bfloat16* gate_tile =
        reinterpret_cast<const __nv_bfloat16*>(shared + kBufferOffset + buffer * kBufferBytes);
    const __nv_bfloat16* value_tile = gate_tile + kTileRows * kTileWidth;
    uint4 gate_raw[kMaxItems], value_raw[kMaxItems];
#pragma unroll
    for (int i = 0; i < kMaxItems; ++i) {
      const int row = first_row + kRowStride * i;
      if (active && row < kTileRows) {
        gate_raw[i] = *reinterpret_cast<const uint4*>(gate_tile + swizzled(row, group) / 2);
        value_raw[i] = *reinterpret_cast<const uint4*>(value_tile + swizzled(row, group) / 2);
      }
    }
    __syncwarp();
    if (lane == 0) mbarrier_arrive(buffer_empty + 8 * buffer);
#pragma unroll
    for (int i = 0; i < kMaxItems; ++i) {
      const int row = first_row + kRowStride * i;
      const int64_t global_row = row0 + row;
      if (!active || row >= kTileRows || global_row >= p.rows || col >= p.width) continue;
      float gate[8], value[8], hidden[8];
      unpack8(gate_raw[i], gate);
      unpack8(value_raw[i], value);
      const uint32_t keep = keep8(p.drop_threshold, stream, global_row * row_groups + col / 8);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        hidden[e] = gate[e] * sigmoid(gate[e]) * value[e] * ((keep >> e) & 1u ? p.keep_scale : 0.f);
      }
      *reinterpret_cast<uint4*>(p.out + global_row * p.width + col) = pack8(hidden);
    }
  }
}

// Rows per block of the backward, walked by kBiasLanes row lanes: the block's column sums are one bias partial.
constexpr int kBiasRows = 32;
constexpr int kBiasLanes = 4;

// The gradient of [gate | value], [rows, 2 width] bf16, and per block of kBiasRows rows its fp32 column sums, the bias
// gradient's partials, in a fixed order: per lane by row, then by lane. One thread per (row, 8 columns), the forward's
// dropout draw.
__global__ void __launch_bounds__(1024)
    swiglu_backward_bias_kernel(const __nv_bfloat16* __restrict__ grad, const __nv_bfloat16* __restrict__ gate_value,
                                __nv_bfloat16* __restrict__ grad_gate_value, float* __restrict__ bias_partial,
                                int64_t rows, int width, const int64_t* rng, uint32_t threshold, float keep_scale) {
  extern __shared__ float lane_sums[];  // [kBiasLanes - 1][blockDim.x][16]
  const int col = threadIdx.x * 8;
  const bool active = col < width;
  const int64_t row0 = static_cast<int64_t>(blockIdx.x) * kBiasRows;
  const PhiloxStream stream = philox_stream(rng);
  float gate_sum[8] = {}, value_sum[8] = {};
#pragma unroll 2
  for (int r = threadIdx.y; r < kBiasRows; r += kBiasLanes) {
    const int64_t row = row0 + r;
    if (!active || row >= rows) break;
    const __nv_bfloat16* source = gate_value + row * 2 * width;
    float gate[8], value[8], upstream[8], grad_gate[8], grad_value[8];
    unpack8(*reinterpret_cast<const uint4*>(source + col), gate);
    unpack8(*reinterpret_cast<const uint4*>(source + width + col), value);
    unpack8(*reinterpret_cast<const uint4*>(grad + row * width + col), upstream);
    const uint32_t keep = keep8(threshold, stream, row * (width / 8) + col / 8);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float g = upstream[i] * ((keep >> i) & 1u ? keep_scale : 0.f);
      const float s = sigmoid(gate[i]);
      grad_value[i] = g * gate[i] * s;
      grad_gate[i] = g * value[i] * s * (1.f + gate[i] * (1.f - s));
    }
    const uint4 packed_gate = pack8(grad_gate), packed_value = pack8(grad_value);
    __nv_bfloat16* target = grad_gate_value + row * 2 * width;
    *reinterpret_cast<uint4*>(target + col) = packed_gate;
    *reinterpret_cast<uint4*>(target + width + col) = packed_value;
    // The bias gradient sums the bf16 gradients, as the reference's reduction does.
    unpack8(packed_gate, grad_gate);
    unpack8(packed_value, grad_value);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      gate_sum[i] += grad_gate[i];
      value_sum[i] += grad_value[i];
    }
  }
  if (threadIdx.y > 0) {
    float* slot = lane_sums + ((threadIdx.y - 1) * blockDim.x + threadIdx.x) * 16;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      slot[i] = gate_sum[i];
      slot[8 + i] = value_sum[i];
    }
  }
  __syncthreads();
  if (threadIdx.y > 0 || !active) return;
  for (int lane = 1; lane < kBiasLanes; ++lane) {
    const float* slot = lane_sums + ((lane - 1) * blockDim.x + threadIdx.x) * 16;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      gate_sum[i] += slot[i];
      value_sum[i] += slot[8 + i];
    }
  }
  float* partial = bias_partial + static_cast<int64_t>(blockIdx.x) * 2 * width;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    partial[col + i] = gate_sum[i];
    partial[width + col + i] = value_sum[i];
  }
}

// The fused feed-forward backward: tiles of 128 rows x 64 hidden columns of grad_hidden = grad_out w2 (K = 256) by the
// MMA warpgroups into a shared buffer as bf16, as the reference's GEMM rounds it; the epilogue warps turn each tile
// into the [gate | value] gradient with the forward's dropout draws and its fp32 column sums, the bias gradient's
// partials.
constexpr int kBwdTileWidth = 64;
constexpr int kBwdAccumulators = kBwdTileWidth / 2;
constexpr int kBwdBBlockBytes = kBwdTileWidth * kBlockK * 2;
constexpr int kBwdStageOffset = kMaxKBlocks * kBwdBBlockBytes;
constexpr int kBwdBufferBytes = kTileRows * kBwdTileWidth * 2;
constexpr int kBwdBufferOffset = kBwdStageOffset + kStages * kABytes;
// Per buffer, the forward's gate and value tiles of the tile, staged by the load warp for the epilogue.
constexpr int kBwdGateValueOffset = kBwdBufferOffset + kBuffers * kBwdBufferBytes;
constexpr int kBwdBarrierOffset = kBwdGateValueOffset + kBuffers * 2 * kBwdBufferBytes;
constexpr int kBwdBarriers = 2 * kStages + 4 * kBuffers + 2;
// One MMA warpgroup multiplies both 64-row halves of a tile, which leaves 11 warps to the epilogue, the bottleneck.
constexpr int kBwdMmaWarps = 4;
constexpr int kBwdEpilogueWarps = 11;
constexpr int kBwdItemThreads = kBwdEpilogueWarps * 32;
constexpr int kBwdRowStride = kBwdItemThreads / kGroups;
constexpr int kBwdMaxItems = (kTileRows + kBwdRowStride - 1) / kBwdRowStride;
static_assert(kBwdMmaWarps + kBwdEpilogueWarps + 1 == kThreads / 32, "the backward keeps the forward's block size");
// The epilogue warps' per-tile column sums: [warps][groups][8 gate, 8 value].
constexpr int kBwdReduceOffset = kBwdBarrierOffset + 8 * kBwdBarriers;
constexpr int kBwdSharedBytes = kBwdReduceOffset + kBwdEpilogueWarps * kGroups * 16 * 4 + 1024;

struct BackwardGemmParams {
  int rows, width, row_tiles, tiles;
  const __nv_bfloat16* gate_value;  // [rows, 2 width], the forward's
  __nv_bfloat16* grad_gate_value;   // [rows, 2 width]
  float* bias_partial;              // [row_tiles, 2 width], zeros but at each block's column-tile segment starts
  const int64_t* rng;               // the forward's dropout stream
  uint32_t drop_threshold;          // 0 keeps everything
  float keep_scale;
};

__device__ __forceinline__ void wgmma_fence_tile(float (&d)[kBwdAccumulators]) {
#pragma unroll
  for (int i = 0; i < kBwdAccumulators; ++i) asm volatile("" : "+f"(d[i])::"memory");
  PY123D_SM90A(asm volatile("wgmma.fence.sync.aligned;\n" :: : "memory"));
}

template <int N>
__device__ __forceinline__ void wgmma_wait_tile(float (&d)[kBwdAccumulators]) {
  PY123D_SM90A(asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"));
#pragma unroll
  for (int i = 0; i < kBwdAccumulators; ++i) asm volatile("" : "+f"(d[i])::"memory");
}

// d (+)= a b^T for a 64 x 16 tile of a and a 64 x 16 tile of b, both K-major in shared memory.
__device__ __forceinline__ void wgmma_m64n64(float (&d)[kBwdAccumulators], uint64_t a, uint64_t b, bool accumulate) {
  PY123D_SM90A(asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\n"
      "}\n" : "+f"(d[0]),
      "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]),
      "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
      "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]),
      "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b),
      "r"(static_cast<uint32_t>(accumulate))));
}

// The warp's 16 rows of an m64n64 accumulator tile as bf16 into a swizzled 64-column tile at rows [row0, +16), laid out
// like store_half's halves.
__device__ __forceinline__ void store_tile(const float (&acc)[kBwdAccumulators], uint32_t tile, int row0, int lane) {
  const int matrix = lane / 8, row = row0 + lane % 8 + 8 * (matrix % 2);
#pragma unroll
  for (int j = 0; j < kGroups; j += 2) {
    const float* a = acc + 4 * j;
    stmatrix_x4(tile + swizzled(row, j + matrix / 2), pack_bf16(a[0], a[1]), pack_bf16(a[2], a[3]),
                pack_bf16(a[4], a[5]), pack_bf16(a[6], a[7]));
  }
}

__device__ __forceinline__ void epilogue_barrier() {
  asm volatile("bar.sync 2, %0;\n" ::"n"(kBwdEpilogueWarps * 32) : "memory");
}

// Blocks take contiguous ranges of the [column tiles x row tiles] tiles, row tiles fastest, so a block keeps one
// weight tile in shared memory over many row tiles. The load warp streams grad_out's K blocks and the forward's gate
// and value tiles; the MMA warpgroup multiplies and hands bf16 grad_hidden tiles to the epilogue warps through two
// buffers.
__global__ void __launch_bounds__(kThreads, 1)
    swiglu_backward_gemm_kernel(const __grid_constant__ CUtensorMap grad_map, const __grid_constant__ CUtensorMap w_map,
                                const __grid_constant__ CUtensorMap gate_value_map, const BackwardGemmParams p) {
  extern __shared__ unsigned char shared_raw[];
  const uint32_t raw_address = shared_address(shared_raw);
  const uint32_t base = (raw_address + 1023u) & ~1023u;
  unsigned char* shared = shared_raw + (base - raw_address);
  const uint32_t full = base + kBwdBarrierOffset, empty = full + 8 * kStages;
  const uint32_t buffer_full = empty + 8 * kStages, buffer_empty = buffer_full + 8 * kBuffers;
  const uint32_t gate_value_full = buffer_empty + 8 * kBuffers, gate_value_empty = gate_value_full + 8 * kBuffers;
  const uint32_t weights_full = gate_value_empty + 8 * kBuffers, weights_empty = weights_full + 8;
  float* reduce = reinterpret_cast<float*>(shared + kBwdReduceOffset);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    for (int stage = 0; stage < kStages; ++stage) {
      mbarrier_init(full + 8 * stage, 1);
      mbarrier_init(empty + 8 * stage, kBwdMmaWarps);
    }
    for (int buffer = 0; buffer < kBuffers; ++buffer) {
      mbarrier_init(buffer_full + 8 * buffer, kBwdMmaWarps);
      mbarrier_init(buffer_empty + 8 * buffer, kBwdEpilogueWarps);
      mbarrier_init(gate_value_full + 8 * buffer, 1);
      mbarrier_init(gate_value_empty + 8 * buffer, kBwdEpilogueWarps);
    }
    mbarrier_init(weights_full, 1);
    mbarrier_init(weights_empty, kBwdMmaWarps);
    PY123D_SM90A(asm volatile("fence.mbarrier_init.release.cluster;\n" :: : "memory"));
  }
  __syncthreads();
  const int first = static_cast<int>(static_cast<int64_t>(blockIdx.x) * p.tiles / gridDim.x);
  const int last = static_cast<int>(static_cast<int64_t>(blockIdx.x + 1) * p.tiles / gridDim.x);
  if (warp == kBwdMmaWarps + kBwdEpilogueWarps) {
    if (lane != 0) return;
    int stage = 0, column_tile = -1, columns = 0;
    uint32_t phase = 0;
    for (int tile = first; tile < last; ++tile) {
      const int tile_column = tile / p.row_tiles, row0 = tile % p.row_tiles * kTileRows;
      if (tile_column != column_tile) {
        // The MMA warpgroups are done with the previous weight tile.
        if (column_tile >= 0) mbarrier_wait(weights_empty, (columns - 1) & 1);
        column_tile = tile_column;
        ++columns;
        mbarrier_expect(weights_full, kMaxKBlocks * kBwdBBlockBytes);
        for (int k_block = 0; k_block < kMaxKBlocks; ++k_block) {
          tma_load_2d(base + k_block * kBwdBBlockBytes, &w_map, k_block * kBlockK, column_tile * kBwdTileWidth,
                      weights_full);
        }
      }
      for (int k_block = 0; k_block < kMaxKBlocks; ++k_block) {
        mbarrier_wait(empty + 8 * stage, phase ^ 1);
        mbarrier_expect(full + 8 * stage, kABytes);
        tma_load_2d(base + kBwdStageOffset + stage * kABytes, &grad_map, k_block * kBlockK, row0, full + 8 * stage);
        if (++stage == kStages) {
          stage = 0;
          phase ^= 1;
        }
      }
    }
    return;
  }

  if (warp < kBwdMmaWarps) {
    const int warp_row0 = warp * 16;
    float acc[2][kBwdAccumulators];
    int stage = 0, column_tile = -1, columns = 0;
    uint32_t phase = 0;
    for (int tile = first; tile < last; ++tile) {
      const int tile_column = tile / p.row_tiles;
      if (tile_column != column_tile) {
        column_tile = tile_column;
        mbarrier_wait(weights_full, columns & 1);
        ++columns;
      }
      int previous = 0;
      for (int k_block = 0; k_block < kMaxKBlocks; ++k_block) {
        mbarrier_wait(full + 8 * stage, phase);
        wgmma_fence_tile(acc[0]);
        wgmma_fence_tile(acc[1]);
        const uint32_t a = base + kBwdStageOffset + stage * kABytes, b = base + k_block * kBwdBBlockBytes;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
          for (int k = 0; k < kBlockK / 16; ++k) {
            wgmma_m64n64(acc[half], matrix_descriptor(a + half * 64 * 128 + 32 * k), matrix_descriptor(b + 32 * k),
                         k_block > 0 || k > 0);
          }
        }
        wgmma_commit();
        if (k_block > 0) {
          wgmma_wait_tile<1>(acc[0]);
          wgmma_wait_tile<1>(acc[1]);
          if (lane == 0) mbarrier_arrive(empty + 8 * previous);
        }
        previous = stage;
        if (++stage == kStages) {
          stage = 0;
          phase ^= 1;
        }
      }
      wgmma_wait_tile<0>(acc[0]);
      wgmma_wait_tile<0>(acc[1]);
      if (lane == 0) {
        mbarrier_arrive(empty + 8 * previous);
        if (tile + 1 < last && (tile + 1) / p.row_tiles != tile_column) mbarrier_arrive(weights_empty);
      }
      const int count = tile - first, buffer = count % kBuffers;
      mbarrier_wait(buffer_empty + 8 * buffer, ((count / kBuffers) & 1) ^ 1);
      const uint32_t tile_address = base + kBwdBufferOffset + buffer * kBwdBufferBytes;
      store_tile(acc[0], tile_address, warp_row0, lane);
      store_tile(acc[1], tile_address, 64 + warp_row0, lane);
      __syncwarp();
      if (lane == 0) mbarrier_arrive(buffer_full + 8 * buffer);
    }
    return;
  }

  const int thread = threadIdx.x - kBwdMmaWarps * 32, group = thread % kGroups, first_row = thread / kGroups;
  const int epilogue_warp = warp - kBwdMmaWarps;
  const PhiloxStream stream = philox_stream(p.rng);
  const int64_t row_stride = 2 * static_cast<int64_t>(p.width);
  const int64_t row_groups = p.width / 8;
  // The epilogue's thread 0 loads the forward's gate and value tiles one tile ahead, so the load warp only feeds the
  // MMA warpgroup.
  auto load_gate_value = [&](int tile) {
    const int buffer = (tile - first) % kBuffers;
    const uint32_t gate_tile = base + kBwdGateValueOffset + buffer * 2 * kBwdBufferBytes;
    const int row0 = tile % p.row_tiles * kTileRows, column0 = tile / p.row_tiles * kBwdTileWidth;
    mbarrier_expect(gate_value_full + 8 * buffer, 2 * kBwdBufferBytes);
    tma_load_3d(gate_tile, &gate_value_map, column0, 0, row0, gate_value_full + 8 * buffer);
    tma_load_3d(gate_tile + kBwdBufferBytes, &gate_value_map, column0, 1, row0, gate_value_full + 8 * buffer);
  };
  if (thread == 0 && first < last) load_gate_value(first);
  // Column sums of the block's consecutive tiles of one column tile; written to the bias partial row of the segment's
  // first row tile when the column tile changes.
  float gate_sum[8] = {}, value_sum[8] = {};
  int segment_row_tile = first < last ? first % p.row_tiles : 0;
  for (int tile = first; tile < last; ++tile) {
    const int count = tile - first, buffer = count % kBuffers;
    if (thread == 0 && tile + 1 < last) {
      // The next tile's buffer held the previous tile's.
      if (count >= 1) mbarrier_wait(gate_value_empty + 8 * ((count + 1) % kBuffers), ((count - 1) / kBuffers) & 1);
      load_gate_value(tile + 1);
    }
    const int row0 = tile % p.row_tiles * kTileRows;
    const int column_tile = tile / p.row_tiles, col = column_tile * kBwdTileWidth + 8 * group;
    mbarrier_wait(gate_value_full + 8 * buffer, (count / kBuffers) & 1);
    mbarrier_wait(buffer_full + 8 * buffer, (count / kBuffers) & 1);
    const __nv_bfloat16* grad_tile =
        reinterpret_cast<const __nv_bfloat16*>(shared + kBwdBufferOffset + buffer * kBwdBufferBytes);
    const __nv_bfloat16* gate_tile =
        reinterpret_cast<const __nv_bfloat16*>(shared + kBwdGateValueOffset + buffer * 2 * kBwdBufferBytes);
    const __nv_bfloat16* value_tile = gate_tile + kTileRows * kBwdTileWidth;
    // Branch-free over the items, so their dropout draws and activations interleave; invalid items store nothing.
#pragma unroll
    for (int i = 0; i < kBwdMaxItems; ++i) {
      const int row = first_row + kBwdRowStride * i;
      const int64_t global_row = row0 + row;
      const bool valid = row < kTileRows && global_row < p.rows && col < p.width;
      const uint32_t offset = swizzled(row & (kTileRows - 1), group) / 2;
      float gate[8], value[8], upstream[8], grad_gate[8], grad_value[8];
      unpack8(*reinterpret_cast<const uint4*>(gate_tile + offset), gate);
      unpack8(*reinterpret_cast<const uint4*>(value_tile + offset), value);
      unpack8(*reinterpret_cast<const uint4*>(grad_tile + offset), upstream);
      const uint32_t keep = keep8(p.drop_threshold, stream, global_row * row_groups + col / 8);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float g = upstream[e] * ((keep >> e) & 1u ? p.keep_scale : 0.f);
        const float s = sigmoid(gate[e]);
        grad_value[e] = g * gate[e] * s;
        grad_gate[e] = g * value[e] * s * (1.f + gate[e] * (1.f - s));
      }
      const uint4 packed_gate = pack8(grad_gate), packed_value = pack8(grad_value);
      if (valid) {
        *reinterpret_cast<uint4*>(p.grad_gate_value + global_row * row_stride + col) = packed_gate;
        *reinterpret_cast<uint4*>(p.grad_gate_value + global_row * row_stride + p.width + col) = packed_value;
      }
      // The bias gradient sums the bf16 gradients, as the reference's reduction does.
      unpack8(packed_gate, grad_gate);
      unpack8(packed_value, grad_value);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        gate_sum[e] += valid ? grad_gate[e] : 0.f;
        value_sum[e] += valid ? grad_value[e] : 0.f;
      }
    }
    __syncwarp();
    if (lane == 0) {
      mbarrier_arrive(buffer_empty + 8 * buffer);
      mbarrier_arrive(gate_value_empty + 8 * buffer);
    }
    if (tile + 1 < last && (tile + 1) / p.row_tiles == column_tile) continue;
    // The segment's column sums: lanes of a warp with equal group, then the epilogue warps in order.
#pragma unroll
    for (int e = 0; e < 8; ++e) {
#pragma unroll
      for (int offset = 8; offset <= 16; offset <<= 1) {
        gate_sum[e] += __shfl_xor_sync(0xffffffffu, gate_sum[e], offset);
        value_sum[e] += __shfl_xor_sync(0xffffffffu, value_sum[e], offset);
      }
    }
    if (lane < kGroups) {
      float* slot = reduce + (epilogue_warp * kGroups + lane) * 16;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        slot[e] = gate_sum[e];
        slot[8 + e] = value_sum[e];
      }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) gate_sum[e] = value_sum[e] = 0.f;
    epilogue_barrier();
    if (thread < 2 * kBwdTileWidth) {
      const int half = thread / kBwdTileWidth, column = thread % kBwdTileWidth;
      const int global_column = column_tile * kBwdTileWidth + column;
      float total = 0.f;
#pragma unroll
      for (int w = 0; w < kBwdEpilogueWarps; ++w)
        total += reduce[(w * kGroups + column / 8) * 16 + half * 8 + column % 8];
      if (global_column < p.width) {
        p.bias_partial[static_cast<int64_t>(segment_row_tile) * row_stride + half * p.width + global_column] = total;
      }
    }
    epilogue_barrier();
    segment_row_tile = (tile + 1) % p.row_tiles;
  }
}

// A tensor map of the bf16 tensor with the given dims (innermost first), byte strides of the outer dims and box; reads
// outside the tensor return zeros and stores outside it are dropped.
CUtensorMap tensor_map(const at::Tensor& tensor, int rank, const cuuint64_t* dims, const cuuint64_t* strides,
                       const cuuint32_t* box, bool swizzle) {
  return encode_tiled(tensor, rank, dims, strides, box,
                      swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, "planner_swiglu_linear");
}

struct Gemm {
  CUtensorMap x_map, w_map;
  GemmParams params;
  int grid;
};

// Threads per row of the backward along the columns: 8 columns each, rounded up to whole warps.
int chunk_threads(int width) { return (width / 8 + 31) / 32 * 32; }

// The feed-forward's first projection and activation: dropout(silu(gate) * value) of [gate | value] = x weight^T +
// bias. Returns the activation, [gate | value] [rows, 2 width] and the dropout stream's (seed, offset) for the
// backward.
std::tuple<at::Tensor, at::Tensor, at::Tensor> planner_swiglu_linear(const at::Tensor& x, const at::Tensor& weight,
                                                                     const at::Tensor& bias, double dropout_p) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
                  bias.scalar_type() == at::kBFloat16,
              "planner_swiglu_linear runs in bf16 on CUDA");
  const auto properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 9 && properties->minor == 0, "planner_swiglu_linear needs an sm_90 GPU");
  TORCH_CHECK(weight.is_contiguous() && bias.is_contiguous(), "weight and bias must be contiguous");
  const auto x_rows = x.reshape({-1, x.size(-1)}).contiguous();
  const int64_t rows = x_rows.size(0), in_features = x_rows.size(1);
  TORCH_CHECK(in_features % kBlockK == 0 && in_features <= kMaxKBlocks * kBlockK,
              "planner_swiglu_linear needs in_features a multiple of ", kBlockK, " up to ", kMaxKBlocks * kBlockK);
  TORCH_CHECK(
      weight.dim() == 2 && weight.size(1) == in_features && weight.size(0) % 16 == 0 && bias.numel() == weight.size(0),
      "weight must be [2 width, in_features] with width a multiple of 8, bias [2 width]");
  const int64_t width = weight.size(0) / 2;
  auto sizes = x.sizes().vec();
  sizes.back() = width;
  auto hidden = at::empty(sizes, x.options());
  auto gate_value = at::empty({rows, 2 * width}, x.options());
  auto rng = at::empty({2}, x.options().dtype(at::kLong));
  GemmParams p{};
  p.rows = static_cast<int>(rows);
  p.width = static_cast<int>(width);
  p.k_blocks = static_cast<int>(in_features / kBlockK);
  p.row_tiles = static_cast<int>((rows + kTileRows - 1) / kTileRows);
  p.tiles = p.row_tiles * static_cast<int>((width + kTileWidth - 1) / kTileWidth);
  p.bias = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr());
  p.drop_threshold = drop_threshold(dropout_p);
  p.keep_scale = static_cast<float>(1.0 / (1.0 - dropout_p));
  p.philox = p.drop_threshold != 0u ? philox_state() : at::PhiloxCudaState(0, 0);
  p.rng = rng.data_ptr<int64_t>();
  p.out = reinterpret_cast<__nv_bfloat16*>(hidden.data_ptr());
  if (p.tiles == 0) {
    rng.zero_();
    return {hidden, gate_value, rng};
  }
  const cuuint64_t x_dims[2] = {static_cast<cuuint64_t>(in_features), static_cast<cuuint64_t>(rows)};
  const cuuint64_t x_strides[1] = {static_cast<cuuint64_t>(in_features * 2)};
  const cuuint32_t x_box[2] = {kBlockK, kTileRows};
  const CUtensorMap x_map = tensor_map(x_rows, 2, x_dims, x_strides, x_box, true);
  const cuuint64_t w_dims[3] = {static_cast<cuuint64_t>(in_features), static_cast<cuuint64_t>(width), 2};
  const cuuint64_t w_strides[2] = {static_cast<cuuint64_t>(in_features * 2),
                                   static_cast<cuuint64_t>(width * in_features * 2)};
  const cuuint32_t w_box[3] = {kBlockK, kTileWidth, 1};
  const CUtensorMap w_map = tensor_map(weight, 3, w_dims, w_strides, w_box, true);
  const cuuint64_t gv_dims[3] = {static_cast<cuuint64_t>(width), 2, static_cast<cuuint64_t>(rows)};
  const cuuint64_t gv_strides[2] = {static_cast<cuuint64_t>(width * 2), static_cast<cuuint64_t>(width * 4)};
  const cuuint32_t gv_box[3] = {kTileWidth, 1, kTileRows};
  const CUtensorMap gate_value_map = tensor_map(gate_value, 3, gv_dims, gv_strides, gv_box, true);
  static const bool configured = [] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(swiglu_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes));
    return true;
  }();
  (void)configured;
  swiglu_gemm_kernel<<<std::min(p.tiles, properties->multiProcessorCount), kThreads, kSharedBytes,
                       at::cuda::getCurrentCUDAStream()>>>(x_map, w_map, gate_value_map, p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {hidden, gate_value, rng};
}

// Gradients of x, weight and bias; the bias gradient's column sums come out of the activation's backward pass.
std::tuple<at::Tensor, at::Tensor, at::Tensor> planner_swiglu_linear_backward(const at::Tensor& grad,
                                                                              const at::Tensor& x,
                                                                              const at::Tensor& weight,
                                                                              const at::Tensor& gate_value,
                                                                              double dropout_p, const at::Tensor& rng) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(gate_value.is_cuda() && gate_value.scalar_type() == at::kBFloat16 && gate_value.is_contiguous() &&
                  gate_value.size(-1) % 16 == 0,
              "gate_value must be the forward's");
  TORCH_CHECK(rng.is_cuda() && rng.scalar_type() == at::kLong && rng.numel() == 2, "rng must be the forward's");
  const int width = static_cast<int>(gate_value.size(-1) / 2);
  const int64_t rows = gate_value.size(0);
  const auto grad_rows = grad.reshape({rows, width}).contiguous();
  auto grad_gate_value = at::empty_like(gate_value);
  const int64_t row_blocks = (rows + kBiasRows - 1) / kBiasRows;
  auto partial = at::empty({row_blocks, 2 * width}, gate_value.options().dtype(at::kFloat));
  const dim3 block(chunk_threads(width), kBiasLanes);
  TORCH_CHECK(block.x * block.y <= 1024, "planner_swiglu_linear supports hidden widths up to 2048");
  const size_t shared = (kBiasLanes - 1) * block.x * 16 * sizeof(float);
  if (rows > 0) {
    swiglu_backward_bias_kernel<<<static_cast<unsigned int>(row_blocks), block, shared,
                                  at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_rows.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gate_value.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(grad_gate_value.data_ptr()), partial.data_ptr<float>(), rows, width,
        rng.data_ptr<int64_t>(), drop_threshold(dropout_p), static_cast<float>(1.0 / (1.0 - dropout_p)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const auto bias_sum = column_sum(partial);
  const auto x_rows = x.reshape({rows, x.size(-1)});
  auto grad_x = at::mm(grad_gate_value, weight).view(x.sizes());
  auto grad_weight = at::mm(grad_gate_value.t(), x_rows);
  return {grad_x, grad_weight, bias_sum.to(at::kBFloat16)};
}

// The feed-forward: dropout(silu(gate) * value) of [gate | value] = x w1^T + b1, then out = hidden w2^T + b2. Returns
// out, the activation, [gate | value] and the dropout stream for the backward.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> planner_swiglu_feed_forward(
    const at::Tensor& x, const at::Tensor& w1, const at::Tensor& b1, const at::Tensor& w2, const at::Tensor& b2,
    double dropout_p) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(w2.scalar_type() == at::kBFloat16 && b2.scalar_type() == at::kBFloat16 && w2.is_contiguous(),
              "w2 and b2 must be bf16, w2 contiguous");
  auto [hidden, gate_value, rng] = planner_swiglu_linear(x, w1, b1, dropout_p);
  TORCH_CHECK(w2.dim() == 2 && w2.size(1) == hidden.size(-1) && b2.numel() == w2.size(0),
              "w2 must be [out_features, width] and b2 [out_features]");
  auto sizes = x.sizes().vec();
  sizes.back() = w2.size(0);
  auto out = at::addmm(b2, hidden.reshape({-1, hidden.size(-1)}), w2.t()).view(sizes);
  return {out, hidden, gate_value, rng};
}

// Gradients of x, w1, b1, w2 and b2: grad_hidden and the [gate | value] gradient in one fused GEMM, then cuBLAS.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> planner_swiglu_feed_forward_backward(
    const at::Tensor& grad, const at::Tensor& x, const at::Tensor& w1, const at::Tensor& w2, const at::Tensor& hidden,
    const at::Tensor& gate_value, double dropout_p, const at::Tensor& rng) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(gate_value.is_cuda() && gate_value.scalar_type() == at::kBFloat16 && gate_value.is_contiguous() &&
                  gate_value.size(-1) % 16 == 0,
              "gate_value must be the forward's");
  TORCH_CHECK(rng.is_cuda() && rng.scalar_type() == at::kLong && rng.numel() == 2, "rng must be the forward's");
  const auto properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 9 && properties->minor == 0, "planner_swiglu_feed_forward needs an sm_90 GPU");
  const int width = static_cast<int>(gate_value.size(-1) / 2);
  const int64_t rows = gate_value.size(0), out_features = w2.size(0);
  TORCH_CHECK(out_features == kMaxKBlocks * kBlockK, "planner_swiglu_feed_forward_backward needs out_features ",
              kMaxKBlocks * kBlockK);
  const auto grad_rows = grad.reshape({rows, out_features}).contiguous();
  const auto w2_t = w2.t().contiguous();
  auto grad_gate_value = at::empty_like(gate_value);
  BackwardGemmParams p{};
  p.rows = static_cast<int>(rows);
  p.width = width;
  p.row_tiles = static_cast<int>((rows + kTileRows - 1) / kTileRows);
  p.tiles = p.row_tiles * ((width + kBwdTileWidth - 1) / kBwdTileWidth);
  // Each block writes one row per column-tile segment, at the segment's first row tile; the other rows stay zero.
  auto partial = at::zeros({p.row_tiles, 2 * width}, gate_value.options().dtype(at::kFloat));
  p.gate_value = reinterpret_cast<const __nv_bfloat16*>(gate_value.data_ptr());
  p.grad_gate_value = reinterpret_cast<__nv_bfloat16*>(grad_gate_value.data_ptr());
  p.bias_partial = partial.data_ptr<float>();
  p.rng = rng.data_ptr<int64_t>();
  p.drop_threshold = drop_threshold(dropout_p);
  p.keep_scale = static_cast<float>(1.0 / (1.0 - dropout_p));
  if (p.tiles > 0) {
    const cuuint64_t grad_dims[2] = {static_cast<cuuint64_t>(out_features), static_cast<cuuint64_t>(rows)};
    const cuuint64_t grad_strides[1] = {static_cast<cuuint64_t>(out_features * 2)};
    const cuuint32_t grad_box[2] = {kBlockK, kTileRows};
    const CUtensorMap grad_map = tensor_map(grad_rows, 2, grad_dims, grad_strides, grad_box, true);
    const cuuint64_t w_dims[2] = {static_cast<cuuint64_t>(out_features), static_cast<cuuint64_t>(width)};
    const cuuint64_t w_strides[1] = {static_cast<cuuint64_t>(out_features * 2)};
    const cuuint32_t w_box[2] = {kBlockK, kBwdTileWidth};
    const CUtensorMap w_map = tensor_map(w2_t, 2, w_dims, w_strides, w_box, true);
    const cuuint64_t gv_dims[3] = {static_cast<cuuint64_t>(width), 2, static_cast<cuuint64_t>(rows)};
    const cuuint64_t gv_strides[2] = {static_cast<cuuint64_t>(width * 2), static_cast<cuuint64_t>(width * 4)};
    const cuuint32_t gv_box[3] = {kBwdTileWidth, 1, kTileRows};
    const CUtensorMap gate_value_map = tensor_map(gate_value, 3, gv_dims, gv_strides, gv_box, true);
    static const bool configured = [] {
      C10_CUDA_CHECK(cudaFuncSetAttribute(swiglu_backward_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          kBwdSharedBytes));
      return true;
    }();
    (void)configured;
    swiglu_backward_gemm_kernel<<<std::min(p.tiles, properties->multiProcessorCount), kThreads, kBwdSharedBytes,
                                  at::cuda::getCurrentCUDAStream()>>>(grad_map, w_map, gate_value_map, p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const auto bias_sum = column_sum(partial);
  const auto x_rows = x.reshape({rows, x.size(-1)});
  auto grad_x = at::mm(grad_gate_value, w1).view(x.sizes());
  auto grad_w1 = at::mm(grad_gate_value.t(), x_rows);
  auto grad_w2 = at::mm(grad_rows.t(), hidden.reshape({rows, hidden.size(-1)}));
  auto grad_b2 = grad_rows.sum(0);
  return {grad_x, grad_w1, bias_sum.to(at::kBFloat16), grad_w2, grad_b2};
}
}  // namespace

TORCH_LIBRARY_IMPL(py123d_garage_cuda, CUDA, m) {
  m.impl("planner_swiglu_linear", &planner_swiglu_linear);
  m.impl("planner_swiglu_linear_backward", &planner_swiglu_linear_backward);
  m.impl("planner_swiglu_feed_forward", &planner_swiglu_feed_forward);
  m.impl("planner_swiglu_feed_forward_backward", &planner_swiglu_feed_forward_backward);
}

}  // namespace py123d_garage_cuda
