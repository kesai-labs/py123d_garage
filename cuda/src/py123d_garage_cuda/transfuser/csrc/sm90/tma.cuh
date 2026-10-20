// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// The sm_90 instructions the kernels issue by hand, mbarrier and the TMA's tensor and bulk copies, and the driver's
// tensor-map encoder the host calls.

#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "transfuser/csrc/arch.cuh"
#include "transfuser/csrc/sm80/ptx.cuh"

namespace py123d_garage_cuda {

__device__ __forceinline__ void mbarrier_init(uint32_t barrier, int count) {
  PY123D_SM90(asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(barrier), "r"(count)));
}

// Makes the barriers this thread initialized visible to the TMA; once per block, after the inits.
__device__ __forceinline__ void fence_mbarrier_init() {
  PY123D_SM90(asm volatile("fence.mbarrier_init.release.cluster;\n" :: : "memory"));
}

__device__ __forceinline__ void mbarrier_expect(uint32_t barrier, int bytes) {
  PY123D_SM90(
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(barrier), "r"(bytes) : "memory"));
}

__device__ __forceinline__ void mbarrier_wait(uint32_t barrier, int parity) {
  PY123D_SM90(
      asm volatile("{\n"
                   ".reg .pred done;\n"
                   "WAIT_%=:\n"
                   "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
                   "@!done bra WAIT_%=;\n"
                   "}\n" ::"r"(barrier),
                   "r"(parity) : "memory"));
}

// The box of `map` at coordinates (c, w, h, n) into shared memory, completing on `barrier`; zeros outside the tensor.
__device__ __forceinline__ void tma_load(uint32_t destination, const CUtensorMap* map, int c, int w, int h, int n,
                                         uint32_t barrier) {
  PY123D_SM90(asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], "
      "[%6];\n" ::"r"(destination),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c), "r"(w), "r"(h), "r"(n), "r"(barrier) : "memory"));
}

// Shared memory into the box of `map` at (c, w, h, n); the parts outside the tensor are dropped.
__device__ __forceinline__ void tma_store(const CUtensorMap* map, int c, int w, int h, int n, uint32_t source) {
  PY123D_SM90(
      asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, {%1, %2, %3, %4}], [%5];\n" ::"l"(
                       reinterpret_cast<uint64_t>(map)),
                   "r"(c), "r"(w), "r"(h), "r"(n), "r"(source) : "memory"));
}

__device__ __forceinline__ void tma_store_commit() {
  PY123D_SM90(asm volatile("cp.async.bulk.commit_group;\n" :: : "memory"));
}

// Waits until at most N committed store groups still read shared memory.
template <int N>
__device__ __forceinline__ void tma_store_wait_read() {
  PY123D_SM90(asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory"));
}

// Waits until at most N committed store groups are still in flight.
template <int N>
__device__ __forceinline__ void tma_store_wait() {
  PY123D_SM90(asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(N) : "memory"));
}

// A bulk copy of `bytes` (a multiple of 16) into shared memory, completing on `barrier`.
__device__ __forceinline__ void bulk_load(uint32_t destination, const void* source, int bytes, uint32_t barrier) {
  PY123D_SM90(asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(destination),
      "l"(source), "r"(bytes), "r"(barrier) : "memory"));
}

// One bulk copy of 128 bytes from shared to global memory in the group tma_store_commit closes.
__device__ __forceinline__ void bulk_store_128(void* global, const void* shared) {
  PY123D_SM90(asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 128;\n" ::"l"(global),
                           "r"(shared_address(shared)) : "memory"));
}

// Makes this thread's shared-memory writes visible to the TMA.
__device__ __forceinline__ void fence_async_shared() {
  PY123D_SM90(asm volatile("fence.proxy.async.shared::cta;\n" :: : "memory"));
}

// The tensor map of `tensor` with `rank` dimensions, through the driver entry point of this CUDA runtime. The driver
// call needs the device's primary context bound to this thread, which an autograd thread may not have yet.
inline CUtensorMap encode_tiled(const at::Tensor& tensor, cuuint32_t rank, const cuuint64_t* dims,
                                const cuuint64_t* strides, const cuuint32_t* box, CUtensorMapSwizzle swizzle,
                                CUtensorMapL2promotion promotion, const char* op) {
  using EncodeTiled = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  static const EncodeTiled encode = [] {
    void* function = nullptr;
    cudaDriverEntryPointQueryResult status;
#if CUDART_VERSION >= 12050
    C10_CUDA_CHECK(
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &function, 12000, cudaEnableDefault, &status));
#else
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &function, cudaEnableDefault, &status));
#endif
    TORCH_CHECK(status == cudaDriverEntryPointSuccess && function != nullptr,
                "no cuTensorMapEncodeTiled in this driver");
    return reinterpret_cast<EncodeTiled>(function);
  }();
  C10_CUDA_CHECK(cudaSetDevice(tensor.get_device()));
  const cuuint32_t element_strides[5] = {1, 1, 1, 1, 1};
  CUtensorMap map;
  const CUresult result =
      encode(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, tensor.data_ptr(), dims, strides, box, element_strides,
             CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, promotion, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, op, ": cuTensorMapEncodeTiled failed with ", static_cast<int>(result));
  return map;
}

}  // namespace py123d_garage_cuda
