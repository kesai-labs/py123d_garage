// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// The op schemas of the kernels any sm_80 or newer GPU runs.

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(py123d_garage_cuda, m) {
  m.def("fusion_upsample_add(Tensor x, Tensor grid) -> Tensor");
  m.def("fusion_upsample_add_backward(Tensor grad, int grid_height, int grid_width) -> Tensor");
  m.def("resnet_stem_pool(Tensor y, Tensor values, Tensor weight, Tensor bias) -> (Tensor, Tensor)");
  m.def(
      "resnet_stem_pool_backward(Tensor grad, Tensor y, Tensor offsets, Tensor values, Tensor weight) -> "
      "(Tensor, Tensor, Tensor)");
  // Draws its dropout mask from the CUDA generator and returns the (seed, offset) the backward redraws it from.
  m.def(
      "planner_adaptive_norm(Tensor x, Tensor? branch, Tensor? gate, Tensor? shift, Tensor? scale, Tensor weight, "
      "Tensor bias, float dropout_p, float eps) -> (Tensor, Tensor, Tensor, Tensor)",
      {at::Tag::nondeterministic_seeded});
  m.def(
      "planner_adaptive_norm_backward(Tensor? grad_x_out, Tensor grad_y, Tensor normed, Tensor stats, Tensor? branch, "
      "Tensor? gate, Tensor? shift, Tensor? scale, Tensor weight, Tensor bias, float dropout_p, Tensor rng, float eps) "
      "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("depthwise_conv(Tensor x, Tensor weight, Tensor? bias) -> Tensor");
  m.def("depthwise_conv_backward(Tensor grad, Tensor x, Tensor weight) -> (Tensor, Tensor, Tensor)");
  // Draws its dropout mask from the CUDA generator and returns it as keep bits for the backward.
  m.def(
      "planner_attention(Tensor q_src, Tensor kv_src, int q_col, int k_col, int v_col, int heads, Tensor q_weight, "
      "Tensor q_bias, Tensor k_weight, Tensor k_bias, Tensor? mask, float dropout_p, float eps, bool shared) -> "
      "(Tensor, Tensor, Tensor)",
      {at::Tag::nondeterministic_seeded});
  m.def(
      "planner_attention_backward(Tensor grad, Tensor q_src, Tensor kv_src, int q_col, int k_col, int v_col, "
      "int heads, Tensor q_weight, Tensor q_bias, Tensor k_weight, Tensor k_bias, Tensor? mask, float dropout_p, "
      "Tensor keep_bits, float eps, Tensor out, Tensor lse, bool shared) -> (Tensor, Tensor, Tensor, Tensor, Tensor, "
      "Tensor)");
  m.def("planner_dropout_keep(Tensor keep_bits, int batch, int heads, int queries, int keys, Tensor? mask) -> Tensor");
  m.def("planner_attention_backward_shared_bytes(int queries, int keys, bool dropout, bool mask) -> int");
}
