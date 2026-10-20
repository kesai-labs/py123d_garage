// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// The 3 x 3, stride-1, padding-1 convolution between 64 channels (ResNet's first stage), its input gradient and its
// weight gradient, on channels-last bf16 tensors with fp32 accumulation; sm_90a only (TMA, wgmma).
//
// An implicit GEMM with the 64 output channels as M and a tile of 16 x 8 output pixels as N: one TMA box brings the
// tile's 18 x 10 input pixel halo into shared memory in the 128-byte swizzle, and every tap's B operand is a view into
// it, shifted by the tap. wgmma applies the swizzle to the absolute shared-memory address, so the view may start at any
// pixel and its 8-pixel rows lie a halo row (10 pixels) apart. The halo is read once for all nine taps, where an
// im2col load per tap reads every input pixel nine times. The input gradient is the same GEMM over the gradient's halo
// with the flipped taps' weight tiles read transposed. The weight gradient sums over pixels: per tap row, M = the 64
// output channels of the output gradient tile, N = the three taps' 64 input channels, three halo views one pixel apart
// read as one 192-wide operand; persistent blocks keep the weight gradient in registers and write fp32 partial sums,
// which a second kernel adds in a fixed order.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

#include "transfuser/csrc/sm90/tma.cuh"
#include "transfuser/csrc/sm90a/resnet_halo_conv.h"

namespace py123d_garage_cuda {
namespace {

constexpr int kChannels = 64;
constexpr int kTileRows = 16;
constexpr int kTileCols = 8;
constexpr int kHaloRows = kTileRows + 2;
constexpr int kHaloCols = kTileCols + 2;
constexpr int kHaloBytes = kHaloRows * kHaloCols * kChannels * 2;
// Stages start on the swizzle's 1024-byte pattern.
constexpr int kStageBytes = (kHaloBytes + 1023) / 1024 * 1024;
constexpr int kStages = 4;
constexpr int kTapBytes = kChannels * kChannels * 2;
constexpr int kWeightBytes = 9 * kTapBytes;
constexpr int kOutBytes = kTileRows * kTileCols * kChannels * 2;
constexpr int kConsumers = 2;
constexpr int kThreads = 128 * (1 + kConsumers);
constexpr int kSmem = 1024 + kWeightBytes + kStages * kStageBytes + kConsumers * kOutBytes + (2 * kStages + 1) * 8;
// The weight gradient's tiles are 32 x 8 pixels; its stages hold a halo and the output gradient tile, and its
// consumers take one tap row each.
constexpr int kWgradRows = 32;
constexpr int kWgradHaloBytes = (kWgradRows + 2) * kHaloCols * kChannels * 2;
constexpr int kWgradHaloStageBytes = (kWgradHaloBytes + 1023) / 1024 * 1024;
constexpr int kGradBytes = kWgradRows * kTileCols * kChannels * 2;
constexpr int kWgradStageBytes = kWgradHaloStageBytes + kGradBytes;
constexpr int kWgradStages = 3;
// Blocks get about kWgradTilesPerBlock tiles, at least kWgradMinBlocks blocks: every block adds 147 KB of fp32 partial
// sums, and next to the other seeds' kernels fewer, longer blocks cost less (measured with three concurrent copies).
constexpr int kWgradTilesPerBlock = 24;
constexpr int kWgradMinBlocks = 32;
constexpr int kWgradThreads = 128 * 4;
constexpr int kWgradSmem = 1024 + kWgradStages * kWgradStageBytes + 2 * kWgradStages * 8;
// Elements of the [64 k, 3, 3, 64 c] weight gradient.
constexpr int kWeightElements = 9 * kChannels * kChannels;

struct Geometry {
  int tiles_w, tiles_hw, num_tiles;
};

#if defined(__CUDA_ARCH_FEAT_SM90_ALL)

__device__ __forceinline__ uint32_t smem_addr(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(count));
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_addr(bar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred done;\nWAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      "@!done bra WAIT_%=;\n}\n" ::"r"(smem_addr(bar)),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void tma_load_4d(void* dst, const CUtensorMap* map, int c, int w, int h, int n,
                                            uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], "
      "[%6];\n" ::"r"(smem_addr(dst)),
      "l"(map), "r"(c), "r"(w), "r"(h), "r"(n), "r"(smem_addr(bar))
      : "memory");
}

__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, int x, int y, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], "
      "[%4];\n" ::"r"(smem_addr(dst)),
      "l"(map), "r"(x), "r"(y), "r"(smem_addr(bar))
      : "memory");
}

__device__ __forceinline__ void tma_store_4d(const CUtensorMap* map, const void* src, int c, int w, int h, int n) {
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%1, %2, %3, %4}], [%5];\n" ::"l"(map),
               "r"(c), "r"(w), "r"(h), "r"(n), "r"(smem_addr(src))
               : "memory");
}

__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }

__device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }

__device__ __forceinline__ void bulk_wait() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }

__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

__device__ __forceinline__ void named_barrier(int id, int threads) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(threads) : "memory");
}

// An operand in the 128-byte swizzle with 8-row groups stride_bytes apart.
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr, uint32_t stride_bytes) {
  return static_cast<uint64_t>((addr & 0x3FFFF) >> 4) | (uint64_t{1} << 16) |
         (static_cast<uint64_t>(stride_bytes >> 4) << 32) | (uint64_t{1} << 62);
}

// Keeps the compiler from moving accumulator accesses across the asynchronous wgmma.
template <int N>
__device__ __forceinline__ void fence_operands(float (&d)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(d[i])::"memory");
}

// d (+)= A B for a 64 x 16 A (K-major, or M-major with kTransA) and a K-major 16 x 128 B.
template <int kTransA>
__device__ __forceinline__ void wgmma_m64n128k16(float (&d)[64], uint64_t desc_a, uint64_t desc_b, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, "
      "%46, "
      "%47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, p, 1, 1, %67, "
      "0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),
        "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(kTransA));
}

// d += A B for a 64 x 16 A read M-major and a 16 x 192 B read N-major.
__device__ __forceinline__ void wgmma_m64n192k16_tt(float (&d)[96], uint64_t desc_a, uint64_t desc_b) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, "
      "%24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, "
      "%46, "
      "%47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, "
      "%69, "
      "%70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, "
      "%92, "
      "%93, %94, %95}, %96, %97, p, 1, 1, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]),
        "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]),
        "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]),
        "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]),
        "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),
        "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]),
        "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]),
        "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]),
        "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]),
        "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95])
      : "l"(desc_a), "l"(desc_b));
}

// An operand with leading_bytes between its 64-element atoms along M or N and 8-row groups stride_bytes apart.
__device__ __forceinline__ uint64_t smem_desc_mn(uint32_t addr, uint32_t leading_bytes, uint32_t stride_bytes) {
  return static_cast<uint64_t>((addr & 0x3FFFF) >> 4) | (static_cast<uint64_t>(leading_bytes >> 4) << 16) |
         (static_cast<uint64_t>(stride_bytes >> 4) << 32) | (uint64_t{1} << 62);
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&pair);
}

__device__ __forceinline__ void stmatrix_x4_trans(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), "r"(r0), "r"(r1),
               "r"(r2), "r"(r3)
               : "memory");
}

#endif

// y = conv(x, weight), or with kDataGrad y = conv(x, flip(weight)^T) = the gradient of the convolution's input for
// the output gradient x. Persistent blocks take tiles round robin; a producer warp group loads the weight once and the
// halos into a ring of stages, two consumer warp groups take turns on the tiles and store through TMA.
template <bool kDataGrad>
__global__ void __launch_bounds__(kThreads, 1)
    halo_conv_kernel(const __grid_constant__ CUtensorMap map_x, const __grid_constant__ CUtensorMap map_y,
                     const __grid_constant__ CUtensorMap map_w, Geometry geo) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* smem =
      reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t{1023});
  unsigned char* w_s = smem;
  unsigned char* halo_s = w_s + kWeightBytes;
  unsigned char* out_s = halo_s + kStages * kStageBytes;
  uint64_t* full = reinterpret_cast<uint64_t*>(out_s + kConsumers * kOutBytes);
  uint64_t* empty = full + kStages;
  uint64_t* weight_bar = empty + kStages;

  const int tid = threadIdx.x;
  const int wg = tid / 128;
  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], 1);
    }
    mbar_init(weight_bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (tid == 0) {
      // Tap t's [64 k, 64 c] weight tile; the weight is [64, 9 x 64] in the channels-last [K, C, 3, 3] layout.
      mbar_expect_tx(weight_bar, kWeightBytes);
      for (int tap = 0; tap < 9; ++tap) tma_load_2d(w_s + tap * kTapBytes, &map_w, tap * kChannels, 0, weight_bar);
      int i = 0;
      for (int tile = blockIdx.x; tile < geo.num_tiles; tile += gridDim.x, ++i) {
        const int stage = i % kStages;
        const int n = tile / geo.tiles_hw;
        const int rest = tile - n * geo.tiles_hw;
        mbar_wait(&empty[stage], ((i / kStages) & 1) ^ 1);
        mbar_expect_tx(&full[stage], kHaloBytes);
        // Outside the image the box reads zeros: the padding.
        tma_load_4d(halo_s + stage * kStageBytes, &map_x, 0, rest % geo.tiles_w * kTileCols - 1,
                    rest / geo.tiles_w * kTileRows - 1, n, &full[stage]);
      }
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
    const int consumer = wg - 1;
    const int local = tid % 128;
    const int warp = local / 32;
    const int lane = local % 32;
    unsigned char* my_out = out_s + consumer * kOutBytes;
    const uint32_t w_base = smem_addr(w_s);
    const uint32_t out_base = smem_addr(my_out);
    mbar_wait(weight_bar, 0);
    float acc[64];
    int i = 0;
    for (int tile = blockIdx.x; tile < geo.num_tiles; tile += gridDim.x, ++i) {
      if (i % kConsumers != consumer) continue;
      const int stage = i % kStages;
      mbar_wait(&full[stage], (i / kStages) & 1);
      const uint32_t halo_base = smem_addr(halo_s + stage * kStageBytes);
      fence_operands(acc);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
      for (int tap = 0; tap < 9; ++tap) {
        const int r = tap / 3;
        const int s = tap % 3;
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          // The input gradient reads the flipped tap's [k, c] tile transposed: M = c, its K = k rows 128 bytes apart.
          const uint64_t desc_a = kDataGrad ? smem_desc(w_base + (8 - tap) * kTapBytes + kb * 2048, 1024)
                                            : smem_desc(w_base + tap * kTapBytes + kb * 32, 1024);
          const uint64_t desc_b = smem_desc(halo_base + (r * kHaloCols + s) * 128 + kb * 32, kHaloCols * 128);
          wgmma_m64n128k16<kDataGrad ? 1 : 0>(acc, desc_a, desc_b, (tap | kb) != 0);
        }
      }
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
      fence_operands(acc);
      if (local == 0) {
        mbar_arrive(&empty[stage]);
        // This buffer's previous store has read it.
        bulk_wait_read();
      }
      named_barrier(1 + consumer, 128);
      // acc holds output channels as rows and pixels as columns; stmatrix writes [pixel, 64 channels] rows.
      const int matrix = lane / 8;
      const int row = lane % 8;
#pragma unroll
      for (int j = 0; j < 16; j += 2) {
        const int pixel = 8 * (j + matrix / 2) + row;
        const int chunk = 2 * warp + matrix % 2;
        stmatrix_x4_trans(out_base + pixel * 128 + (((chunk ^ pixel) & 7) << 4), pack_bf16(acc[4 * j], acc[4 * j + 1]),
                          pack_bf16(acc[4 * j + 2], acc[4 * j + 3]), pack_bf16(acc[4 * j + 4], acc[4 * j + 5]),
                          pack_bf16(acc[4 * j + 6], acc[4 * j + 7]));
      }
      fence_proxy_async();
      named_barrier(1 + consumer, 128);
      if (local == 0) {
        const int n = tile / geo.tiles_hw;
        const int rest = tile - n * geo.tiles_hw;
        tma_store_4d(&map_y, my_out, 0, rest % geo.tiles_w * kTileCols, rest / geo.tiles_w * kTileRows, n);
        bulk_commit();
      }
    }
    if (local == 0) bulk_wait();
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

// partial[block] = the weight gradient over the block's tiles, fp32 [64 k, 3, 3, 64 c]: sum over the tile pixels of
// grad[pixel, k] x[pixel + tap, c]. A producer warp group loads each tile's input halo and output gradient into a ring
// of stages; consumer warp group r sums tap row r.
__global__ void __launch_bounds__(kWgradThreads, 1)
    halo_wgrad_kernel(const __grid_constant__ CUtensorMap map_x, const __grid_constant__ CUtensorMap map_grad,
                      float* __restrict__ partial, Geometry geo) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  unsigned char* smem =
      reinterpret_cast<unsigned char*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t{1023});
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + kWgradStages * kWgradStageBytes);
  uint64_t* empty = full + kWgradStages;
  const int tid = threadIdx.x;
  const int wg = tid / 128;
  if (tid == 0) {
    for (int s = 0; s < kWgradStages; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], 3);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (wg == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (tid == 0) {
      int i = 0;
      for (int tile = blockIdx.x; tile < geo.num_tiles; tile += gridDim.x, ++i) {
        const int stage = i % kWgradStages;
        const int n = tile / geo.tiles_hw;
        const int rest = tile - n * geo.tiles_hw;
        const int h0 = rest / geo.tiles_w * kWgradRows;
        const int w0 = rest % geo.tiles_w * kTileCols;
        unsigned char* stage_s = smem + stage * kWgradStageBytes;
        mbar_wait(&empty[stage], ((i / kWgradStages) & 1) ^ 1);
        mbar_expect_tx(&full[stage], kWgradHaloBytes + kGradBytes);
        tma_load_4d(stage_s, &map_x, 0, w0 - 1, h0 - 1, n, &full[stage]);
        // Outside the image the gradient reads zeros, so those pixels add nothing.
        tma_load_4d(stage_s + kWgradHaloStageBytes, &map_grad, 0, w0, h0, n, &full[stage]);
      }
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 152;\n");
    const int tap_row = wg - 1;
    const int local = tid % 128;
    float acc[96];
#pragma unroll
    for (int e = 0; e < 96; ++e) acc[e] = 0.0f;
    int i = 0;
    for (int tile = blockIdx.x; tile < geo.num_tiles; tile += gridDim.x, ++i) {
      const int stage = i % kWgradStages;
      mbar_wait(&full[stage], (i / kWgradStages) & 1);
      const uint32_t halo_base = smem_addr(smem + stage * kWgradStageBytes);
      const uint32_t grad_base = halo_base + kWgradHaloStageBytes;
      fence_operands(acc);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
      for (int kb = 0; kb < kWgradRows / 2; ++kb) {
        // Pixels 16 kb .. 16 kb + 15: output rows 2 kb and 2 kb + 1, which read halo rows 2 kb + tap_row on.
        const uint64_t desc_a = smem_desc_mn(grad_base + kb * 2048, 128, 1024);
        const uint64_t desc_b = smem_desc_mn(halo_base + (2 * kb + tap_row) * kHaloCols * 128, 128, kHaloCols * 128);
        wgmma_m64n192k16_tt(acc, desc_a, desc_b);
      }
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
      fence_operands(acc);
      if (local == 0) mbar_arrive(&empty[stage]);
    }
    // acc rows are output channels k, columns tap s x 64 + c of tap row tap_row.
    const int warp = local / 32;
    const int lane = local % 32;
    float* out = partial + static_cast<int64_t>(blockIdx.x) * kWeightElements + tap_row * 3 * kChannels;
#pragma unroll
    for (int j = 0; j < 24; ++j) {
      const int k = 16 * warp + lane / 4;
      const int column = 8 * j + 2 * (lane % 4);
      *reinterpret_cast<float2*>(out + k * 9 * kChannels + column) = make_float2(acc[4 * j], acc[4 * j + 1]);
      *reinterpret_cast<float2*>(out + (k + 8) * 9 * kChannels + column) = make_float2(acc[4 * j + 2], acc[4 * j + 3]);
    }
  }
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

constexpr int kReduceThreads = 256;

// grad_weight[e] = the bf16 sum of partial[b, e] over the blocks b in order.
__global__ void __launch_bounds__(kReduceThreads)
    halo_wgrad_reduce_kernel(const float* __restrict__ partial, __nv_bfloat16* __restrict__ grad_weight, int blocks) {
  const int e = blockIdx.x * kReduceThreads + threadIdx.x;
  if (e >= kWeightElements) return;
  float total = 0.0f;
#pragma unroll 8
  for (int b = 0; b < blocks; ++b) total += partial[static_cast<int64_t>(b) * kWeightElements + e];
  grad_weight[e] = __float2bfloat16(total);
}

CUtensorMap encode(const at::Tensor& t, int rank, const cuuint64_t* dims, const cuuint64_t* strides,
                   const cuuint32_t* box) {
  return encode_tiled(t, rank, dims, strides, box, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                      "resnet_halo_conv");
}

// The map of a channels-last [N, 64, H, W] tensor with boxes of box_h x box_w pixels.
CUtensorMap activation_map(const at::Tensor& t, int box_w, int box_h) {
  const cuuint64_t dims[4] = {kChannels, static_cast<cuuint64_t>(t.size(3)), static_cast<cuuint64_t>(t.size(2)),
                              static_cast<cuuint64_t>(t.size(0))};
  const cuuint64_t strides[3] = {static_cast<cuuint64_t>(t.stride(3) * 2), static_cast<cuuint64_t>(t.stride(2) * 2),
                                 static_cast<cuuint64_t>(t.stride(0) * 2)};
  const cuuint32_t box[4] = {kChannels, static_cast<cuuint32_t>(box_w), static_cast<cuuint32_t>(box_h), 1};
  return encode(t, 4, dims, strides, box);
}

template <bool kDataGrad>
void launch(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& y) {
  Geometry geo;
  geo.tiles_w = static_cast<int>((x.size(3) + kTileCols - 1) / kTileCols);
  geo.tiles_hw = static_cast<int>((x.size(2) + kTileRows - 1) / kTileRows) * geo.tiles_w;
  geo.num_tiles = geo.tiles_hw * static_cast<int>(x.size(0));
  const int sms = at::cuda::getDeviceProperties(x.device().index())->multiProcessorCount;
  auto kernel = halo_conv_kernel<kDataGrad>;
  static const bool configured = [&] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    return true;
  }();
  (void)configured;
  const cuuint64_t weight_dims[2] = {9 * kChannels, kChannels};
  const cuuint64_t weight_strides[1] = {9 * kChannels * 2};
  const cuuint32_t weight_box[2] = {kChannels, kChannels};
  const CUtensorMap map_x = activation_map(x, kHaloCols, kHaloRows);
  const CUtensorMap map_y = activation_map(y, kTileCols, kTileRows);
  const CUtensorMap map_w = encode(weight, 2, weight_dims, weight_strides, weight_box);
  kernel<<<std::min(sms, geo.num_tiles), kThreads, kSmem, at::cuda::getCurrentCUDAStream()>>>(map_x, map_y, map_w, geo);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

at::Tensor halo_conv_weight_grad(const at::Tensor& grad, const at::Tensor& x) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(grad.data_ptr()) % 16 == 0,
      "resnet_halo_conv: tensors must be 16-byte aligned");
  Geometry geo;
  geo.tiles_w = static_cast<int>((x.size(3) + kTileCols - 1) / kTileCols);
  geo.tiles_hw = static_cast<int>((x.size(2) + kWgradRows - 1) / kWgradRows) * geo.tiles_w;
  geo.num_tiles = geo.tiles_hw * static_cast<int>(x.size(0));
  const int sms = at::cuda::getDeviceProperties(x.device().index())->multiProcessorCount;
  const int blocks = std::min({sms, geo.num_tiles, std::max(kWgradMinBlocks, geo.num_tiles / kWgradTilesPerBlock)});
  static const bool configured = [] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(halo_wgrad_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kWgradSmem));
    return true;
  }();
  (void)configured;
  auto partial = at::empty({blocks, kWeightElements}, x.options().dtype(at::kFloat));
  const CUtensorMap map_x = activation_map(x, kHaloCols, kWgradRows + 2);
  const CUtensorMap map_grad = activation_map(grad, kTileCols, kWgradRows);
  const auto stream = at::cuda::getCurrentCUDAStream();
  halo_wgrad_kernel<<<blocks, kWgradThreads, kWgradSmem, stream>>>(map_x, map_grad, partial.data_ptr<float>(), geo);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto grad_weight = at::empty({kChannels, kChannels, 3, 3}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
  halo_wgrad_reduce_kernel<<<(kWeightElements + kReduceThreads - 1) / kReduceThreads, kReduceThreads, 0, stream>>>(
      partial.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(grad_weight.data_ptr()), blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_weight;
}

bool halo_conv_supports(const at::Tensor& x, const at::Tensor& weight, at::IntArrayRef stride,
                        at::IntArrayRef padding) {
  const auto* properties = at::cuda::getDeviceProperties(x.device().index());
  return properties->major == 9 && properties->minor == 0 && x.dim() == 4 && x.size(1) == kChannels &&
         weight.dim() == 4 && weight.size(0) == kChannels && weight.size(1) == kChannels && weight.size(2) == 3 &&
         weight.size(3) == 3 && stride[0] == 1 && stride[1] == 1 && padding[0] == 1 && padding[1] == 1 &&
         x.numel() < (int64_t{1} << 31);
}

at::Tensor halo_conv(const at::Tensor& x, const at::Tensor& weight, bool data_grad) {
  const c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
      "resnet_halo_conv: tensors must be 16-byte aligned");
  auto y = at::empty({x.size(0), kChannels, x.size(2), x.size(3)},
                     x.options().memory_format(at::MemoryFormat::ChannelsLast));
  if (data_grad) {
    launch<true>(x, weight, y);
  } else {
    launch<false>(x, weight, y);
  }
  return y;
}

}  // namespace py123d_garage_cuda
