// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 KE:SAI

// 2D convolutions (forward, data gradient, weight gradient) on channels-last bf16 tensors with fp32 accumulation,
// each pass on a pinned engine: a cuDNN engine configuration (its global index and knob choices; -1 takes cuDNN's
// first heuristic choice) or, for a 64-channel 3 x 3 stride-1 convolution, the halo kernels of halo_conv.cu (-2).
// cuDNN execution plans are built on the first call of a shape and cached, so a CUDA graph recording reuses them.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cudnn/Handle.h>
#include <c10/cuda/CUDAGuard.h>
#include <cudnn.h>
#include <torch/library.h>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "transfuser/csrc/sm90a/resnet_halo_conv.h"

namespace py123d_garage_cuda {
namespace {

#define CUDNN_BACKEND_CHECK(expr)                                                                              \
  do {                                                                                                         \
    cudnnStatus_t status_ = (expr);                                                                            \
    TORCH_CHECK(status_ == CUDNN_STATUS_SUCCESS, "cuDNN error ", cudnnGetErrorString(status_), " in ", #expr); \
  } while (0)

enum Pass : int64_t { kForward = 0, kDataGrad = 1, kWeightGrad = 2 };

constexpr int64_t kHaloEngine = -2;

// One backend descriptor, destroyed with its owner.
class Descriptor {
 public:
  explicit Descriptor(cudnnBackendDescriptorType_t type) {
    CUDNN_BACKEND_CHECK(cudnnBackendCreateDescriptor(type, &raw_));
  }
  ~Descriptor() { cudnnBackendDestroyDescriptor(raw_); }
  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;

  template <typename T>
  Descriptor& set(cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t type, int64_t count, const T* values) {
    CUDNN_BACKEND_CHECK(cudnnBackendSetAttribute(raw_, name, type, count, static_cast<const void*>(values)));
    return *this;
  }

  cudnnStatus_t finalize() { return cudnnBackendFinalize(raw_); }
  cudnnBackendDescriptor_t raw() const { return raw_; }

 private:
  cudnnBackendDescriptor_t raw_ = nullptr;
};

using DescriptorPtr = std::shared_ptr<Descriptor>;

DescriptorPtr make_descriptor(cudnnBackendDescriptorType_t type) { return std::make_shared<Descriptor>(type); }

// A channels-last bf16 tensor of logical size [n, c, h, w].
DescriptorPtr tensor_descriptor(int64_t uid, std::array<int64_t, 4> size) {
  const std::array<int64_t, 4> strides = {size[1] * size[2] * size[3], 1, size[3] * size[1], size[1]};
  const cudnnDataType_t dtype = CUDNN_DATA_BFLOAT16;
  const int64_t alignment = 16;
  auto descriptor = make_descriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  descriptor->set(CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype)
      .set(CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, size.data())
      .set(CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides.data())
      .set(CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid)
      .set(CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
  CUDNN_BACKEND_CHECK(descriptor->finalize());
  return descriptor;
}

struct Problem {
  int64_t pass;
  std::array<int64_t, 4> input;   // n, c, h, w
  std::array<int64_t, 4> weight;  // k, c, r, s
  std::array<int64_t, 4> output;  // n, k, p, q
  std::array<int64_t, 2> stride;
  std::array<int64_t, 2> padding;
};

struct Plan {
  // The descriptors the plan was built from stay alive with it.
  std::vector<DescriptorPtr> parts;
  DescriptorPtr plan;
  int64_t workspace_bytes = 0;
};

using PlanKey = std::tuple<int, int64_t, std::array<int64_t, 4>, std::array<int64_t, 4>, std::array<int64_t, 2>,
                           std::array<int64_t, 2>, int64_t, std::vector<int64_t>>;

// The uids of x, w and y in every pass.
constexpr int64_t kUidX = 1;
constexpr int64_t kUidW = 2;
constexpr int64_t kUidY = 3;

std::shared_ptr<Plan> build_plan(cudnnHandle_t handle, const Problem& problem, int64_t engine,
                                 const std::vector<int64_t>& knobs) {
  auto result = std::make_shared<Plan>();
  auto x = tensor_descriptor(kUidX, problem.input);
  auto w = tensor_descriptor(kUidW, problem.weight);
  auto y = tensor_descriptor(kUidY, problem.output);

  const int64_t spatial_dims = 2;
  const cudnnDataType_t compute_type = CUDNN_DATA_FLOAT;
  const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  const std::array<int64_t, 2> dilation = {1, 1};
  auto convolution = make_descriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
  convolution->set(CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial_dims)
      .set(CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compute_type)
      .set(CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode)
      .set(CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, dilation.data())
      .set(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, problem.stride.data())
      .set(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, problem.padding.data())
      .set(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, problem.padding.data());
  CUDNN_BACKEND_CHECK(convolution->finalize());

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudnnBackendDescriptor_t x_raw = x->raw();
  cudnnBackendDescriptor_t w_raw = w->raw();
  cudnnBackendDescriptor_t y_raw = y->raw();
  cudnnBackendDescriptor_t convolution_raw = convolution->raw();
  DescriptorPtr operation;
  if (problem.pass == kForward) {
    operation = make_descriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    operation->set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x_raw)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w_raw)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y_raw)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convolution_raw)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta);
  } else if (problem.pass == kDataGrad) {
    operation = make_descriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
    operation->set(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x_raw)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w_raw)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y_raw)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convolution_raw)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, CUDNN_TYPE_FLOAT, 1, &beta);
  } else {
    operation = make_descriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
    operation->set(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x_raw)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w_raw)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y_raw)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convolution_raw)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha)
        .set(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, CUDNN_TYPE_FLOAT, 1, &beta);
  }
  CUDNN_BACKEND_CHECK(operation->finalize());

  cudnnBackendDescriptor_t operation_raw = operation->raw();
  auto graph = make_descriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
  graph->set(CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &operation_raw)
      .set(CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
  CUDNN_BACKEND_CHECK(graph->finalize());
  cudnnBackendDescriptor_t graph_raw = graph->raw();

  DescriptorPtr config;
  if (engine < 0) {
    auto heuristics = make_descriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    const cudnnBackendHeurMode_t heuristic_mode = CUDNN_HEUR_MODE_A;
    heuristics->set(CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph_raw)
        .set(CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &heuristic_mode);
    CUDNN_BACKEND_CHECK(heuristics->finalize());
    config = make_descriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    cudnnBackendDescriptor_t config_raw = config->raw();
    int64_t returned = 0;
    CUDNN_BACKEND_CHECK(cudnnBackendGetAttribute(heuristics->raw(), CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                                 CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &returned,
                                                 static_cast<void*>(&config_raw)));
    TORCH_CHECK(returned == 1, "cuDNN heuristics returned no engine configuration");
    result->parts.push_back(heuristics);
  } else {
    auto engine_descriptor = make_descriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    engine_descriptor->set(CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph_raw)
        .set(CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &engine);
    TORCH_CHECK(engine_descriptor->finalize() == CUDNN_STATUS_SUCCESS, "cuDNN engine ", engine,
                " does not support this convolution");
    std::vector<DescriptorPtr> choices;
    std::vector<cudnnBackendDescriptor_t> choice_raws;
    for (size_t i = 0; i + 1 < knobs.size(); i += 2) {
      const auto knob_type = static_cast<cudnnBackendKnobType_t>(knobs[i]);
      const int64_t knob_value = knobs[i + 1];
      auto choice = make_descriptor(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
      choice->set(CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &knob_type)
          .set(CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, 1, &knob_value);
      CUDNN_BACKEND_CHECK(choice->finalize());
      choice_raws.push_back(choice->raw());
      choices.push_back(choice);
    }
    config = make_descriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    cudnnBackendDescriptor_t engine_raw = engine_descriptor->raw();
    config->set(CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_raw);
    if (!choice_raws.empty()) {
      config->set(CUDNN_ATTR_ENGINECFG_KNOB_CHOICES, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                  static_cast<int64_t>(choice_raws.size()), choice_raws.data());
    }
    TORCH_CHECK(config->finalize() == CUDNN_STATUS_SUCCESS, "cuDNN engine ", engine,
                " rejects its knob choices for this convolution");
    result->parts.push_back(engine_descriptor);
    result->parts.insert(result->parts.end(), choices.begin(), choices.end());
  }

  cudnnBackendDescriptor_t config_raw = config->raw();
  auto plan = make_descriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
  plan->set(CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle)
      .set(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &config_raw);
  TORCH_CHECK(plan->finalize() == CUDNN_STATUS_SUCCESS, "cuDNN engine ", engine,
              " could not build an execution plan for this convolution");
  int64_t returned = 0;
  CUDNN_BACKEND_CHECK(cudnnBackendGetAttribute(plan->raw(), CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64,
                                               1, &returned, &result->workspace_bytes));
  result->parts.insert(result->parts.end(), {x, w, y, convolution, operation, graph, config});
  result->plan = plan;
  return result;
}

std::shared_ptr<Plan> cached_plan(cudnnHandle_t handle, const Problem& problem, int64_t engine,
                                  const std::vector<int64_t>& knobs, int device) {
  static std::mutex mutex;
  static std::map<PlanKey, std::shared_ptr<Plan>> plans;
  PlanKey key{device, problem.pass, problem.input, problem.weight, problem.stride, problem.padding, engine, knobs};
  std::lock_guard<std::mutex> lock(mutex);
  auto found = plans.find(key);
  if (found != plans.end()) {
    return found->second;
  }
  auto plan = build_plan(handle, problem, engine, knobs);
  plans.emplace(key, plan);
  return plan;
}

void execute(const Problem& problem, int64_t engine, const std::vector<int64_t>& knobs, const at::Tensor& x,
             const at::Tensor& w, const at::Tensor& y) {
  const c10::cuda::CUDAGuard guard(x.device());
  cudnnHandle_t handle = at::native::getCudnnHandle();
  CUDNN_BACKEND_CHECK(cudnnSetStream(handle, at::cuda::getCurrentCUDAStream()));
  auto plan = cached_plan(handle, problem, engine, knobs, x.get_device());
  auto workspace = at::empty({std::max<int64_t>(plan->workspace_bytes, 1)}, x.options().dtype(at::kByte));
  std::array<void*, 3> pointers = {x.data_ptr(), w.data_ptr(), y.data_ptr()};
  const std::array<int64_t, 3> uids = {kUidX, kUidW, kUidY};
  void* workspace_pointer = workspace.data_ptr();
  Descriptor pack(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
  pack.set(CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, pointers.data())
      .set(CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids.data())
      .set(CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace_pointer);
  CUDNN_BACKEND_CHECK(pack.finalize());
  CUDNN_BACKEND_CHECK(cudnnBackendExecute(handle, plan->plan->raw(), pack.raw()));
}

std::array<int64_t, 4> sizes4(at::IntArrayRef sizes) { return {sizes[0], sizes[1], sizes[2], sizes[3]}; }

std::array<int64_t, 2> pair(at::IntArrayRef values) { return {values[0], values[1]}; }

void check_input(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == at::kBFloat16 && tensor.dim() == 4, name,
              " must be a 4D bf16 CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(at::MemoryFormat::ChannelsLast), name, " must be channels-last");
}

at::Tensor channels_last_empty(const at::Tensor& like, std::array<int64_t, 4> size) {
  return at::empty({size[0], size[1], size[2], size[3]}, like.options().memory_format(at::MemoryFormat::ChannelsLast));
}

int64_t output_extent(int64_t input, int64_t kernel, int64_t stride, int64_t padding) {
  return (input + 2 * padding - kernel) / stride + 1;
}

at::Tensor pinned_conv(const at::Tensor& x, const at::Tensor& weight, at::IntArrayRef stride, at::IntArrayRef padding,
                       at::IntArrayRef engines, at::IntArrayRef forward_knobs, at::IntArrayRef, at::IntArrayRef) {
  check_input(x, "x");
  check_input(weight, "weight");
  if (engines[0] == kHaloEngine) {
    TORCH_CHECK(halo_conv_supports(x, weight, stride, padding), "pinned_conv: the halo kernel does not run this shape");
    return halo_conv(x, weight, false);
  }
  Problem problem{kForward, sizes4(x.sizes()), sizes4(weight.sizes()), {}, pair(stride), pair(padding)};
  problem.output = {x.size(0), weight.size(0), output_extent(x.size(2), weight.size(2), stride[0], padding[0]),
                    output_extent(x.size(3), weight.size(3), stride[1], padding[1])};
  auto y = channels_last_empty(x, problem.output);
  execute(problem, engines[0], forward_knobs.vec(), x, weight, y);
  return y;
}

at::Tensor pinned_conv_data_grad(const at::Tensor& grad, const at::Tensor& weight, at::IntArrayRef input_size,
                                 at::IntArrayRef stride, at::IntArrayRef padding, int64_t engine,
                                 at::IntArrayRef knobs) {
  check_input(grad, "grad");
  check_input(weight, "weight");
  if (engine == kHaloEngine) {
    TORCH_CHECK(halo_conv_supports(grad, weight, stride, padding) && grad.sizes() == input_size,
                "pinned_conv_data_grad: the halo kernel does not run this shape");
    return halo_conv(grad, weight, true);
  }
  Problem problem{kDataGrad,    sizes4(input_size), sizes4(weight.sizes()), sizes4(grad.sizes()),
                  pair(stride), pair(padding)};
  auto dx = channels_last_empty(grad, problem.input);
  execute(problem, engine, knobs.vec(), dx, weight, grad);
  return dx;
}

at::Tensor pinned_conv_weight_grad(const at::Tensor& grad, const at::Tensor& x, at::IntArrayRef weight_size,
                                   at::IntArrayRef stride, at::IntArrayRef padding, int64_t engine,
                                   at::IntArrayRef knobs) {
  check_input(grad, "grad");
  check_input(x, "x");
  if (engine == kHaloEngine) {
    TORCH_CHECK(x.size(1) == 64 && weight_size[0] == 64 && weight_size[1] == 64 && weight_size[2] == 3 &&
                    weight_size[3] == 3 && stride[0] == 1 && stride[1] == 1 && padding[0] == 1 && padding[1] == 1 &&
                    grad.sizes() == x.sizes(),
                "pinned_conv_weight_grad: the halo kernel does not run this shape");
    return halo_conv_weight_grad(grad, x);
  }
  Problem problem{kWeightGrad,          sizes4(x.sizes()), sizes4(weight_size),
                  sizes4(grad.sizes()), pair(stride),      pair(padding)};
  auto dw = channels_last_empty(grad, problem.weight);
  execute(problem, engine, knobs.vec(), x, dw, grad);
  return dw;
}

}  // namespace

TORCH_LIBRARY_IMPL(py123d_garage_cuda, CUDA, m) {
  m.impl("pinned_conv", &pinned_conv);
  m.impl("pinned_conv_data_grad", &pinned_conv_data_grad);
  m.impl("pinned_conv_weight_grad", &pinned_conv_weight_grad);
}

}  // namespace py123d_garage_cuda
