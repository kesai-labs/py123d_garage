// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// The sm_80 instructions the kernels issue by hand: cp.async, ldmatrix, the bf16 mma.sync shapes and the bf16 pair.

#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace py123d_garage_cuda {

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

// 16 bytes from global to shared memory, zero-filled where not valid, in the group cp_async_commit closes.
__device__ __forceinline__ void cp_async_16(void* shared, const void* global, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(shared_address(shared)), "l"(global),
               "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_async_8(void* shared, const void* global, bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(shared_address(shared)), "l"(global),
               "r"(valid ? 8 : 0));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

// Waits until at most N committed groups are still in flight.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* pointer) {
  ldmatrix_x4(r, shared_address(pointer));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* pointer) {
  ldmatrix_x4_trans(r, shared_address(pointer));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, uint32_t address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(address));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], uint32_t address) { ldmatrix_x2(r[0], r[1], address); }

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1, uint32_t address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(address));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2], uint32_t address) {
  ldmatrix_x2_trans(r[0], r[1], address);
}

// d += a (16 x 16, row) * b (16 x 8, col), bf16 inputs, fp32 accumulators.
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// d += a (16 x 8, row) * b (8 x 8, col), bf16 inputs, fp32 accumulators.
__device__ __forceinline__ void mma_k8(float (&d)[4], const uint32_t (&a)[2], uint32_t b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(b0));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 value = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&value);
}

}  // namespace py123d_garage_cuda
