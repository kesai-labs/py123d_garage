// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// Helpers shared by the planner kernels: Philox4x32-10 streams of the CUDA generator, bf16 packing, the dropout
// threshold and a fixed-order column sum.

#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cuda_bf16.h>

#include <ATen/cuda/PhiloxUtils.cuh>
#include <algorithm>
#include <cstdint>
#include <mutex>

#include "transfuser/csrc/sm80/ptx.cuh"

namespace py123d_garage_cuda {
namespace planner {

// Philox4x32-10: four random words for counter under key.
__device__ __forceinline__ uint4 philox(uint4 counter, uint2 key) {
#pragma unroll
  for (int round = 0; round < 10; ++round) {
    const uint32_t lo0 = 0xD2511F53u * counter.x;
    const uint32_t hi0 = __umulhi(0xD2511F53u, counter.x);
    const uint32_t lo1 = 0xCD9E8D57u * counter.z;
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, counter.z);
    counter = make_uint4(hi1 ^ counter.y ^ key.x, lo1, hi0 ^ counter.w ^ key.y, lo0);
    key.x += 0x9E3779B9u;
    key.y += 0xBB67AE85u;
  }
  return counter;
}

// A call's Philox stream: the key and the high counter words come from the CUDA generator's seed and offset, the low
// counter words number the call's draws, so calls never share a draw.
struct PhiloxStream {
  uint2 key;
  uint2 offset;
};

// Reserves one draw per subsequence of the device's default CUDA generator; graph-safe, so every replay of a recorded
// graph draws anew.
inline at::PhiloxCudaState philox_state() {
  auto* generator =
      at::get_generator_or_default<at::CUDAGeneratorImpl>(std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  const std::lock_guard<std::mutex> lock(generator->mutex_);
  return generator->philox_cuda_state(4);
}

// The stream of a philox_state; writes its (seed, offset) to rng for the backward unless rng is null.
__device__ __forceinline__ PhiloxStream philox_stream(const at::PhiloxCudaState& state, int64_t* rng) {
  const auto [seed, offset] = at::cuda::philox::unpack(state);
  if (rng != nullptr) {
    rng[0] = static_cast<int64_t>(seed);
    rng[1] = static_cast<int64_t>(offset);
  }
  const uint64_t counter = offset / 4;
  return {make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)),
          make_uint2(static_cast<uint32_t>(counter), static_cast<uint32_t>(counter >> 32))};
}

// The stream a forward wrote to rng.
__device__ __forceinline__ PhiloxStream philox_stream(const int64_t* rng) {
  const uint64_t seed = static_cast<uint64_t>(rng[0]), counter = static_cast<uint64_t>(rng[1]) / 4;
  return {make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)),
          make_uint2(static_cast<uint32_t>(counter), static_cast<uint32_t>(counter >> 32))};
}

// Eight 16-bit random numbers of a subsequence, two per word.
__device__ __forceinline__ uint4 philox_draw(const PhiloxStream& stream, uint64_t subsequence) {
  return philox(make_uint4(stream.offset.x, stream.offset.y, static_cast<uint32_t>(subsequence),
                           static_cast<uint32_t>(subsequence >> 32)),
                stream.key);
}

// Keep decisions of a word's two 16-bit numbers: bit 0 the low half, bit 1 the high half.
__device__ __forceinline__ uint32_t keep_pair(uint32_t word, uint32_t threshold) {
  return ((word & 0xffffu) >= threshold ? 1u : 0u) | ((word >> 16) >= threshold ? 2u : 0u);
}

// Keep decisions of the 8 elements of a subsequence's draw, as bits (element e: word e / 2, half e % 2); a zero
// threshold keeps all.
__device__ __forceinline__ uint32_t keep8(uint32_t threshold, const PhiloxStream& stream, uint64_t subsequence) {
  if (threshold == 0u) return 0xffu;
  const uint4 random = philox_draw(stream, subsequence);
  return keep_pair(random.x, threshold) | keep_pair(random.y, threshold) << 2 | keep_pair(random.z, threshold) << 4 |
         keep_pair(random.w, threshold) << 6;
}

__device__ __forceinline__ void unpack8(const uint4& raw, float (&x)[8]) {
  const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 value = __bfloat1622float2(pairs[i]);
    x[2 * i] = value.x;
    x[2 * i + 1] = value.y;
  }
}

__device__ __forceinline__ uint4 pack8(const float (&x)[8]) {
  uint4 raw;
  raw.x = pack_bf16(x[0], x[1]);
  raw.y = pack_bf16(x[2], x[3]);
  raw.z = pack_bf16(x[4], x[5]);
  raw.w = pack_bf16(x[6], x[7]);
  return raw;
}

// The 16-bit Philox number below which an element drops: drops p of them up to 2^-17.
inline uint32_t drop_threshold(double dropout_p) {
  TORCH_CHECK(dropout_p >= 0.0 && dropout_p < 1.0, "dropout_p must lie in [0, 1), got ", dropout_p);
  return static_cast<uint32_t>(std::min(std::nearbyint(dropout_p * 65536.0), 65535.0));
}

constexpr int kColumnSumLanes = 32;

// out[c] = sum over rows of partial[row, c]: blocks of 32 columns x kColumnSumLanes row lanes, each lane summing the
// rows congruent to it in order, then the lanes in order.
static __global__ void column_sum_kernel(const float* partial, int64_t rows, int columns, float* out) {
  __shared__ float lane_sums[kColumnSumLanes][32];
  const int column = blockIdx.x * 32 + threadIdx.x;
  float total = 0.f;
  if (column < columns) {
    for (int64_t row = threadIdx.y; row < rows; row += kColumnSumLanes) total += partial[row * columns + column];
  }
  lane_sums[threadIdx.y][threadIdx.x] = total;
  __syncthreads();
  if (threadIdx.y == 0 && column < columns) {
    float sum = 0.f;
#pragma unroll
    for (int lane = 0; lane < kColumnSumLanes; ++lane) sum += lane_sums[lane][threadIdx.x];
    out[column] = sum;
  }
}

// The column sums of a contiguous fp32 [rows, columns] tensor, in a fixed order.
inline at::Tensor column_sum(const at::Tensor& partial) {
  const int64_t rows = partial.size(0);
  const int columns = static_cast<int>(partial.size(1));
  auto out = at::empty({columns}, partial.options());
  column_sum_kernel<<<(columns + 31) / 32, dim3(32, kColumnSumLanes), 0, at::cuda::getCurrentCUDAStream()>>>(
      partial.data_ptr<float>(), rows, columns, out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace planner
}  // namespace py123d_garage_cuda
