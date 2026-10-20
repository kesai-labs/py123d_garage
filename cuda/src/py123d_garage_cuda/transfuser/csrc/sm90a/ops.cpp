// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// The op schemas of the kernels built on sm_90a instructions, wgmma and setmaxnreg, which only sm_90 GPUs run.

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(py123d_garage_cuda, m) {
  // Each pass on a pinned engine: a cuDNN engine's global index (-1: its first heuristic choice) with [type, value,
  // ...] knobs, or -2 for the halo kernel.
  m.def(
      "pinned_conv(Tensor x, Tensor weight, int[] stride, int[] padding, int[] engines, int[] forward_knobs, "
      "int[] data_knobs, int[] weight_knobs) -> Tensor");
  m.def(
      "pinned_conv_data_grad(Tensor grad, Tensor weight, int[] input_size, int[] stride, int[] padding, int engine, "
      "int[] knobs) -> Tensor");
  m.def(
      "pinned_conv_weight_grad(Tensor grad, Tensor x, int[] weight_size, int[] stride, int[] padding, int engine, "
      "int[] knobs) -> Tensor");
  // Draws its dropout mask from the CUDA generator and returns the (seed, offset) the backward redraws it from.
  m.def("planner_swiglu_linear(Tensor x, Tensor weight, Tensor bias, float dropout_p) -> (Tensor, Tensor, Tensor)",
        {at::Tag::nondeterministic_seeded});
  m.def(
      "planner_swiglu_linear_backward(Tensor grad, Tensor x, Tensor weight, Tensor gate_value, float dropout_p, "
      "Tensor rng) -> (Tensor, Tensor, Tensor)");
  // The feed-forward with its second projection; draws its dropout mask like planner_swiglu_linear.
  m.def(
      "planner_swiglu_feed_forward(Tensor x, Tensor w1, Tensor b1, Tensor w2, Tensor b2, float dropout_p) -> "
      "(Tensor, Tensor, Tensor, Tensor)",
      {at::Tag::nondeterministic_seeded});
  m.def(
      "planner_swiglu_feed_forward_backward(Tensor grad, Tensor x, Tensor w1, Tensor w2, Tensor hidden, "
      "Tensor gate_value, float dropout_p, Tensor rng) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}
