// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// The flow planner's adaptive norm between sublayers, per 256-wide fp32 row: x_out = x + (1 + gate) dropout(branch),
// then y = LN(x_out) (1 + scale) + shift, with gate, shift and scale per batch row. Without a branch it only normalizes
// x, without a scale it skips the modulation. Like the compiled torch reference, 1 + gate and 1 + scale round to bf16
// and everything else stays fp32 until y. The backward reduces the gate, shift and scale gradients over each batch
// row's queries and the norm's weight and bias gradients over all rows, in a fixed order.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/library.h>

#include <cstdint>

#include "transfuser/csrc/sm80/planner.cuh"

namespace py123d_garage_cuda {
namespace {

using namespace planner;

constexpr int kDim = 256;
constexpr int kForwardWarps = 8;
constexpr int kBackwardWarps = 12;
// Per-batch-row sums of the backward: gate, shift, scale, norm weight, norm bias.
constexpr int kSums = 5;

struct Params {
  const float* x;               // [rows, 256]: the input (forward), x_out or the input without a branch (backward)
  const __nv_bfloat16* branch;  // [rows, 256] or nullptr
  const __nv_bfloat16* gate;    // [batch, 256] rows gate_stride apart, or nullptr
  const __nv_bfloat16* shift;
  const __nv_bfloat16* scale;  // nullptr: no modulation
  int64_t gate_stride, shift_stride, scale_stride;
  const float* weight;
  const float* bias;
  int64_t rows;
  int queries;
  float eps;
  uint32_t drop_threshold;  // 0 keeps everything
  float keep_scale;
  at::PhiloxCudaState philox;
  int64_t* rng;  // written by the forward, read by the backward
  float* x_out;
  void* y;        // bf16 with a scale, else fp32
  float2* stats;  // [rows]: mean, rstd
  // Backward.
  const float* grad_x_out;  // or nullptr
  const void* grad_y;
  float* grad_x;
  __nv_bfloat16* grad_branch;
  __nv_bfloat16* grad_gate;  // [batch, 256] each
  __nv_bfloat16* grad_shift;
  __nv_bfloat16* grad_scale;
  float* partial;  // [batch, 2, 256]: the weight and bias gradients' sums per batch row
};

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
}

__device__ __forceinline__ void load8(const float* source, float (&x)[8]) {
  const float4 a = *reinterpret_cast<const float4*>(source), b = *reinterpret_cast<const float4*>(source + 4);
  x[0] = a.x, x[1] = a.y, x[2] = a.z, x[3] = a.w, x[4] = b.x, x[5] = b.y, x[6] = b.z, x[7] = b.w;
}

__device__ __forceinline__ void store8(float* target, const float (&x)[8]) {
  *reinterpret_cast<float4*>(target) = make_float4(x[0], x[1], x[2], x[3]);
  *reinterpret_cast<float4*>(target + 4) = make_float4(x[4], x[5], x[6], x[7]);
}

// 1 + value of 8 bf16 values, rounded to bf16 like the reference's materialized sum.
__device__ __forceinline__ void one_plus8(const __nv_bfloat16* source, float (&x)[8]) {
  unpack8(*reinterpret_cast<const uint4*>(source), x);
#pragma unroll
  for (int e = 0; e < 8; ++e) x[e] = __bfloat162float(__float2bfloat16_rn(1.f + x[e]));
}

// One warp per row, 8 channels per lane; lane l of row r draws the dropout of its channels from subsequence 32 r + l.
template <bool kModulated>
__global__ void __launch_bounds__(kForwardWarps * 32) adaptive_norm_forward_kernel(const Params p) {
  const int lane = threadIdx.x % 32, c0 = lane * 8;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kForwardWarps + threadIdx.x / 32;
  const PhiloxStream stream = philox_stream(p.philox, blockIdx.x == 0 && threadIdx.x == 0 ? p.rng : nullptr);
  if (row >= p.rows) return;
  const int64_t batch = row / p.queries;
  float x[8];
  load8(p.x + row * kDim + c0, x);
  if (p.branch != nullptr) {
    float branch[8], gate[8];
    unpack8(*reinterpret_cast<const uint4*>(p.branch + row * kDim + c0), branch);
    one_plus8(p.gate + batch * p.gate_stride + c0, gate);
    const uint32_t keep = keep8(p.drop_threshold, stream, row * 32 + lane);
#pragma unroll
    for (int e = 0; e < 8; ++e) x[e] += gate[e] * (branch[e] * ((keep >> e) & 1u ? p.keep_scale : 0.f));
    store8(p.x_out + row * kDim + c0, x);
  }
  float sum = 0.f;
#pragma unroll
  for (int e = 0; e < 8; ++e) sum += x[e];
  const float mean = warp_sum(sum) * (1.f / kDim);
  float squares = 0.f;
#pragma unroll
  for (int e = 0; e < 8; ++e) squares += (x[e] - mean) * (x[e] - mean);
  const float rstd = rsqrtf(warp_sum(squares) * (1.f / kDim) + p.eps);
  float weight[8], bias[8], y[8];
  load8(p.weight + c0, weight);
  load8(p.bias + c0, bias);
#pragma unroll
  for (int e = 0; e < 8; ++e) y[e] = (x[e] - mean) * rstd * weight[e] + bias[e];
  if constexpr (kModulated) {
    float scale[8], shift[8];
    one_plus8(p.scale + batch * p.scale_stride + c0, scale);
    unpack8(*reinterpret_cast<const uint4*>(p.shift + batch * p.shift_stride + c0), shift);
#pragma unroll
    for (int e = 0; e < 8; ++e) y[e] = y[e] * scale[e] + shift[e];
    *reinterpret_cast<uint4*>(static_cast<__nv_bfloat16*>(p.y) + row * kDim + c0) = pack8(y);
  } else {
    store8(static_cast<float*>(p.y) + row * kDim + c0, y);
  }
  if (lane == 0) p.stats[row] = make_float2(mean, rstd);
}

// A lane's inputs of one row: x, the gradients of y and x_out, the row's stats, the branch and its dropout keep bits.
struct RowInputs {
  float x[8], grad[8], upstream[8];
  float2 stats;
  uint4 branch;
  uint32_t keep;
};

// One block per batch row, its warps taking the row's queries in turn, 8 channels per lane, each warp loading its next
// row while it computes the current one; the per-warp sums then add up in warp order.
template <bool kModulated>
__global__ void __launch_bounds__(kBackwardWarps * 32) adaptive_norm_backward_kernel(const Params p) {
  extern __shared__ float warp_sums[];  // [warps][kSums][256]
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, c0 = lane * 8;
  const int64_t batch = blockIdx.x;
  const bool residual = p.branch != nullptr;
  PhiloxStream stream{};
  if (residual) stream = philox_stream(p.rng);
  float weight[8], bias[8], scale[8], gate[8];
  load8(p.weight + c0, weight);
  load8(p.bias + c0, bias);
  if (kModulated) one_plus8(p.scale + batch * p.scale_stride + c0, scale);
  if (residual) one_plus8(p.gate + batch * p.gate_stride + c0, gate);
  auto load_row = [&](int64_t row, RowInputs& in) {
    load8(p.x + row * kDim + c0, in.x);
    if constexpr (kModulated) {
      unpack8(*reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(p.grad_y) + row * kDim + c0), in.grad);
    } else {
      load8(static_cast<const float*>(p.grad_y) + row * kDim + c0, in.grad);
    }
    in.stats = p.stats[row];
#pragma unroll
    for (int e = 0; e < 8; ++e) in.upstream[e] = 0.f;
    if (p.grad_x_out != nullptr) load8(p.grad_x_out + row * kDim + c0, in.upstream);
    if (residual) {
      in.branch = *reinterpret_cast<const uint4*>(p.branch + row * kDim + c0);
      in.keep = keep8(p.drop_threshold, stream, row * 32 + lane);
    }
  };
  float sums[kSums][8] = {};
  RowInputs next;
  if (warp < p.queries) load_row(batch * p.queries + warp, next);
  for (int query = warp; query < p.queries; query += kBackwardWarps) {
    const int64_t row = batch * p.queries + query;
    RowInputs in = next;
    if (query + kBackwardWarps < p.queries) load_row(row + kBackwardWarps, next);
    float normalized[8], grad_normalized[8], dot = 0.f, total = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float grad = in.grad[e];
      normalized[e] = (in.x[e] - in.stats.x) * in.stats.y;
      if constexpr (kModulated) {
        sums[1][e] += grad;
        sums[2][e] += grad * (normalized[e] * weight[e] + bias[e]);
        grad *= scale[e];
      }
      sums[3][e] += grad * normalized[e];
      sums[4][e] += grad;
      grad_normalized[e] = grad * weight[e];
      total += grad_normalized[e];
      dot += grad_normalized[e] * normalized[e];
    }
    const float mean_grad = warp_sum(total) * (1.f / kDim), mean_dot = warp_sum(dot) * (1.f / kDim);
    float grad_x[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      grad_x[e] = in.stats.y * (grad_normalized[e] - mean_grad - normalized[e] * mean_dot) + in.upstream[e];
    }
    store8(p.grad_x + row * kDim + c0, grad_x);
    if (residual) {
      float branch[8], grad_branch[8];
      unpack8(in.branch, branch);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float kept = (in.keep >> e) & 1u ? p.keep_scale : 0.f;
        grad_branch[e] = grad_x[e] * gate[e] * kept;
        sums[0][e] += grad_x[e] * branch[e] * kept;
      }
      *reinterpret_cast<uint4*>(p.grad_branch + row * kDim + c0) = pack8(grad_branch);
    }
  }
#pragma unroll
  for (int k = 0; k < kSums; ++k) {
#pragma unroll
    for (int e = 0; e < 8; ++e) warp_sums[(warp * kSums + k) * kDim + c0 + e] = sums[k][e];
  }
  __syncthreads();
  for (int index = threadIdx.x; index < kSums * kDim; index += blockDim.x) {
    const int k = index / kDim, channel = index % kDim;
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kBackwardWarps; ++w) total += warp_sums[(w * kSums + k) * kDim + channel];
    const int64_t at = batch * kDim + channel;
    if (k == 0 && residual) p.grad_gate[at] = __float2bfloat16_rn(total);
    if (k == 1 && kModulated) p.grad_shift[at] = __float2bfloat16_rn(total);
    if (k == 2 && kModulated) p.grad_scale[at] = __float2bfloat16_rn(total);
    if (k >= 3) p.partial[(batch * 2 + k - 3) * kDim + channel] = total;
  }
}

void check_rows(const at::Tensor& tensor, at::ScalarType dtype, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == dtype && tensor.is_contiguous() && tensor.size(-1) == kDim,
              name, " must be a contiguous CUDA ", dtype, " tensor of rows of ", kDim);
}

// A [batch, 256] bf16 per-batch-row tensor with unit channel stride; returns its row stride.
int64_t check_modulation(const at::Tensor& tensor, int64_t batch, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == at::kBFloat16 && tensor.dim() == 2 &&
                  tensor.size(0) == batch && tensor.size(1) == kDim && tensor.stride(1) == 1 &&
                  tensor.stride(0) % 8 == 0,
              name, " must be a CUDA bf16 [batch, ", kDim, "] tensor with 16-byte aligned rows");
  return tensor.stride(0);
}

// The checks and the parameters both directions share.
Params make_params(const at::Tensor& x, const std::optional<at::Tensor>& branch, const std::optional<at::Tensor>& gate,
                   const std::optional<at::Tensor>& shift, const std::optional<at::Tensor>& scale,
                   const at::Tensor& weight, const at::Tensor& bias, double dropout_p, double eps) {
  check_rows(x, at::kFloat, "x");
  TORCH_CHECK(x.dim() == 3, "x must be [batch, queries, ", kDim, "]");
  check_rows(weight, at::kFloat, "weight");
  check_rows(bias, at::kFloat, "bias");
  TORCH_CHECK(weight.numel() == kDim && bias.numel() == kDim, "weight and bias must hold ", kDim, " values");
  TORCH_CHECK(branch.has_value() == gate.has_value(), "branch and gate go together");
  TORCH_CHECK(shift.has_value() == scale.has_value(), "shift and scale go together");
  const int64_t batch = x.size(0);
  Params p{};
  p.x = x.data_ptr<float>();
  p.rows = batch * x.size(1);
  p.queries = static_cast<int>(x.size(1));
  p.weight = weight.data_ptr<float>();
  p.bias = bias.data_ptr<float>();
  p.eps = static_cast<float>(eps);
  if (branch.has_value()) {
    check_rows(*branch, at::kBFloat16, "branch");
    TORCH_CHECK(branch->sizes() == x.sizes(), "branch must have x's shape");
    p.branch = reinterpret_cast<const __nv_bfloat16*>(branch->data_ptr());
    p.gate_stride = check_modulation(*gate, batch, "gate");
    p.gate = reinterpret_cast<const __nv_bfloat16*>(gate->data_ptr());
    p.drop_threshold = drop_threshold(dropout_p);
    p.keep_scale = static_cast<float>(1.0 / (1.0 - dropout_p));
  }
  if (scale.has_value()) {
    p.shift_stride = check_modulation(*shift, batch, "shift");
    p.scale_stride = check_modulation(*scale, batch, "scale");
    p.shift = reinterpret_cast<const __nv_bfloat16*>(shift->data_ptr());
    p.scale = reinterpret_cast<const __nv_bfloat16*>(scale->data_ptr());
  }
  return p;
}

// x_out (empty without a branch), y (bf16 with a scale, else fp32), the rows' (mean, rstd) and the dropout stream's
// (seed, offset).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> planner_adaptive_norm(
    const at::Tensor& x, const std::optional<at::Tensor>& branch, const std::optional<at::Tensor>& gate,
    const std::optional<at::Tensor>& shift, const std::optional<at::Tensor>& scale, const at::Tensor& weight,
    const at::Tensor& bias, double dropout_p, double eps) {
  const c10::cuda::CUDAGuard guard(x.device());
  Params p = make_params(x, branch, gate, shift, scale, weight, bias, dropout_p, eps);
  auto x_out = branch.has_value() ? at::empty_like(x) : x.new_empty({0});
  auto y = at::empty(x.sizes(), x.options().dtype(scale.has_value() ? at::kBFloat16 : at::kFloat));
  auto stats = at::empty({x.size(0), x.size(1), 2}, x.options());
  auto rng = at::empty({2}, x.options().dtype(at::kLong));
  p.philox = p.drop_threshold != 0u ? philox_state() : at::PhiloxCudaState(0, 0);
  p.rng = rng.data_ptr<int64_t>();
  p.x_out = branch.has_value() ? x_out.data_ptr<float>() : nullptr;
  p.y = y.data_ptr();
  p.stats = reinterpret_cast<float2*>(stats.data_ptr<float>());
  if (p.rows == 0) {
    rng.zero_();
    return {x_out, y, stats, rng};
  }
  const auto kernel = scale.has_value() ? adaptive_norm_forward_kernel<true> : adaptive_norm_forward_kernel<false>;
  kernel<<<(p.rows + kForwardWarps - 1) / kForwardWarps, kForwardWarps * 32, 0, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, y, stats, rng};
}

// Gradients of x, branch, gate, shift, scale (empty where absent), weight and bias. normed is the norm's input: x_out
// with a branch, else x.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
planner_adaptive_norm_backward(const std::optional<at::Tensor>& grad_x_out, const at::Tensor& grad_y,
                               const at::Tensor& normed, const at::Tensor& stats,
                               const std::optional<at::Tensor>& branch, const std::optional<at::Tensor>& gate,
                               const std::optional<at::Tensor>& shift, const std::optional<at::Tensor>& scale,
                               const at::Tensor& weight, const at::Tensor& bias, double dropout_p,
                               const at::Tensor& rng, double eps) {
  const c10::cuda::CUDAGuard guard(normed.device());
  Params p = make_params(normed, branch, gate, shift, scale, weight, bias, dropout_p, eps);
  const int64_t batch = normed.size(0);
  const auto grad_y_rows = grad_y.contiguous();
  check_rows(grad_y_rows, scale.has_value() ? at::kBFloat16 : at::kFloat, "grad_y");
  TORCH_CHECK(stats.is_contiguous() && stats.numel() == 2 * p.rows, "stats must be the forward's");
  TORCH_CHECK(rng.is_cuda() && rng.scalar_type() == at::kLong && rng.numel() == 2, "rng must be the forward's");
  at::Tensor grad_x_out_rows;
  if (grad_x_out.has_value()) {
    grad_x_out_rows = grad_x_out->contiguous();
    check_rows(grad_x_out_rows, at::kFloat, "grad_x_out");
    p.grad_x_out = grad_x_out_rows.data_ptr<float>();
  }
  auto grad_x = at::empty_like(normed);
  const auto modulation_options = normed.options().dtype(at::kBFloat16);
  auto grad_branch = branch.has_value() ? at::empty_like(*branch) : normed.new_empty({0}, modulation_options);
  auto grad_gate =
      gate.has_value() ? at::empty({batch, kDim}, modulation_options) : normed.new_empty({0}, modulation_options);
  auto grad_shift =
      shift.has_value() ? at::empty({batch, kDim}, modulation_options) : normed.new_empty({0}, modulation_options);
  auto grad_scale =
      scale.has_value() ? at::empty({batch, kDim}, modulation_options) : normed.new_empty({0}, modulation_options);
  auto partial = at::empty({batch, 2 * kDim}, normed.options());
  p.stats = reinterpret_cast<float2*>(const_cast<float*>(stats.data_ptr<float>()));
  p.rng = const_cast<int64_t*>(rng.data_ptr<int64_t>());
  p.grad_y = grad_y_rows.data_ptr();
  p.grad_x = grad_x.data_ptr<float>();
  p.grad_branch = branch.has_value() ? reinterpret_cast<__nv_bfloat16*>(grad_branch.data_ptr()) : nullptr;
  p.grad_gate = gate.has_value() ? reinterpret_cast<__nv_bfloat16*>(grad_gate.data_ptr()) : nullptr;
  p.grad_shift = shift.has_value() ? reinterpret_cast<__nv_bfloat16*>(grad_shift.data_ptr()) : nullptr;
  p.grad_scale = scale.has_value() ? reinterpret_cast<__nv_bfloat16*>(grad_scale.data_ptr()) : nullptr;
  p.partial = partial.data_ptr<float>();
  if (batch > 0) {
    const auto kernel = scale.has_value() ? adaptive_norm_backward_kernel<true> : adaptive_norm_backward_kernel<false>;
    constexpr int shared_bytes = kBackwardWarps * kSums * kDim * sizeof(float);
    static const bool configured = [&] {
      C10_CUDA_CHECK(cudaFuncSetAttribute(adaptive_norm_backward_kernel<true>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
      C10_CUDA_CHECK(cudaFuncSetAttribute(adaptive_norm_backward_kernel<false>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
      return true;
    }();
    (void)configured;
    kernel<<<batch, kBackwardWarps * 32, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const auto norm_grad = column_sum(partial);
  return {grad_x,
          grad_branch,
          grad_gate,
          grad_shift,
          grad_scale,
          norm_grad.slice(0, 0, kDim).clone(),
          norm_grad.slice(0, kDim, 2 * kDim).clone()};
}

}  // namespace

TORCH_LIBRARY_IMPL(py123d_garage_cuda, CUDA, m) {
  m.impl("planner_adaptive_norm", &planner_adaptive_norm);
  m.impl("planner_adaptive_norm_backward", &planner_adaptive_norm_backward);
}

}  // namespace py123d_garage_cuda
