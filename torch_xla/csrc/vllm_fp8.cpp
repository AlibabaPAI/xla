#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Operators.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/dtype.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/tensor_methods.h"
#include "tsl/platform/human_readable_json.h"
#include "xla/service/custom_call_target_registry.h"

namespace torch_xla {

template <class... Inputs>
inline std::vector<c10::IValue> makeStack(Inputs&&... inputs) {
  return {std::forward<Inputs>(inputs)...};
}

template <class... Args>
inline std::vector<c10::IValue> callOp(const c10::OperatorHandle& op,
                                       Args... args) {
  auto stack = makeStack(std::forward<Args>(args)...);
  op.callBoxed(&stack);
  return stack;
}

std::string XlaTensorsToShapeString(std::vector<XLATensorPtr> xla_tensors,
                                    std::string separator = "|") {
  std::vector<std::string> shape_strs;
  for (auto& t : xla_tensors) {
    std::string shape_str;
    TF_CHECK_OK(tsl::ProtoToHumanReadableJson(t->shape().get().ToProto(),
                                              &shape_str,
                                              /*ignore_accuracy_loss=*/true));
    shape_strs.push_back(shape_str);
  }
  return absl::StrJoin(shape_strs, separator);
}

std::vector<at::Tensor> ShapeStringToTorchTensors(void** buffers,
                                                  std::string shape_str,
                                                  std::string separator = "|") {
  std::vector<std::string> shape_strs = absl::StrSplit(shape_str, separator);
  std::vector<xla::Shape> shapes;
  for (auto& shape : shape_strs) {
    xla::ShapeProto shape_proto;
    TF_CHECK_OK(tsl::HumanReadableJsonToProto(shape, &shape_proto));
    shapes.push_back(xla::Shape(shape_proto));
  }

  std::vector<at::Tensor> torch_tensors;
  auto opts = at::TensorOptions().device(at::kCUDA);
  for (int i = 0; i < shapes.size(); i++) {
    auto shape = shapes[i];
    std::vector<int64_t> strides(shape.dimensions_size());
    int64_t stride = 1;
    for (int dim : shape.layout().minor_to_major()) {
      strides[dim] = stride;
      stride *= shape.dimensions(dim);
    }
    torch_tensors.push_back(at::from_blob(
        buffers[i], {shape.dimensions().begin(), shape.dimensions().end()},
        strides,
        opts.dtype(torch_xla::TorchTypeFromXlaType(shape.element_type()))));
  }
  return torch_tensors;
}

// -------------vllm_dynamic_per_token_scaled_fp8_quant-------------------------

// output: out, scales
// input: input, scale_ub?
std::vector<at::Tensor> vllm_dynamic_per_token_scaled_fp8_quant(
    at::Tensor& out,          // [..., d]
    at::Tensor const& input,  // [..., d]
    at::Tensor& scales, std::optional<at::Tensor> const& scale_ub) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");

  std::vector<at::Tensor> torch_tensors{input};
  if (scale_ub.has_value()) {
    torch_tensors.push_back(scale_ub.value());
  }
  torch_tensors.push_back(out);
  torch_tensors.push_back(scales);

  auto xla_tensors = bridge::GetXlaTensors(torch_tensors);

  std::string opaque = XlaTensorsToShapeString(xla_tensors);

  TF_VLOG(2) << "opaque: " << opaque;

  std::vector<XLATensorPtr> operands;
  operands.push_back(xla_tensors[0]);

  int offset = 0;
  if (scale_ub.has_value()) {
    offset = 1;
    operands.push_back(xla_tensors[1]);
  }

  auto result_shape =
      xla::ShapeUtil::MakeTupleShape({xla_tensors[1 + offset]->shape().get(),
                                      xla_tensors[2 + offset]->shape().get()});

  auto result = tensor_methods::cuda_custom_call(
      operands, 2, result_shape, "vllm_dynamic_per_token_scaled_fp8_quant",
      opaque);
  return bridge::AtenFromXlaTensors(result);
}

// buffers[0]: input
// optional buffers[1]: scale_ub
// buffers[2]: output
// buffers[3]: scales
void custom_call_vllm_dynamic_per_token_scaled_fp8_quant(cudaStream_t stream,
                                                         void** buffers,
                                                         const char* opaque,
                                                         size_t opaque_len) {
  std::string shape_str(opaque, opaque_len);
  auto torch_tensors = ShapeStringToTorchTensors(buffers, shape_str);

  XLA_CHECK(torch_tensors.size() == 3 || torch_tensors.size() == 4);

  std::optional<at::Tensor> scale_ub;

  int offset = 0;
  if (torch_tensors.size() == 4) {
    offset = 1;
    scale_ub = torch_tensors[1];
  }

  static const c10::optional<c10::OperatorHandle> op =
      c10::Dispatcher::singleton().findSchema(
          {"_C::dynamic_per_token_scaled_fp8_quant", ""});
  XLA_CHECK(op.has_value());

  cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream().stream();
  cudaEvent_t event1;
  cudaEventCreateWithFlags(&event1, cudaEventDisableTiming);
  cudaEvent_t event2;
  cudaEventCreateWithFlags(&event2, cudaEventDisableTiming);
  cudaEventRecord(event1, stream);
  cudaStreamWaitEvent(torch_stream, event1);

  // torch.ops._C.dynamic_per_token_scaled_fp8_quant(output, input, scale,
  // scale_ub)
  callOp(*op, torch_tensors[1 + offset], torch_tensors[0],
         torch_tensors[2 + offset], scale_ub);

  cudaEventRecord(event2, torch_stream);
  cudaStreamWaitEvent(stream, event2);
}

// ref: https://gist.github.com/bdhirsh/7dadbf6296f8f7d1abcf4c482f438aaa
void dynamic_per_token_scaled_fp8_quant__functionalization(
    at::Tensor& out, at::Tensor const& input, at::Tensor& scales,
    std::optional<at::Tensor> const& scale_ub) {
  XLA_CHECK(at::functionalization::impl::isFunctionalTensor(out));
  XLA_CHECK(at::functionalization::impl::isFunctionalTensor(input));
  XLA_CHECK(at::functionalization::impl::isFunctionalTensor(scales));

  at::functionalization::impl::sync(out);
  at::functionalization::impl::sync(input);
  at::functionalization::impl::sync(scales);

  auto out_ = at::functionalization::impl::from_functional_tensor(out);
  auto input_ = at::functionalization::impl::from_functional_tensor(input);
  auto scales_ = at::functionalization::impl::from_functional_tensor(scales);
  std::optional<at::Tensor> scale_ub_;
  if (scale_ub.has_value()) {
    XLA_CHECK(
        at::functionalization::impl::isFunctionalTensor(scale_ub.value()));
    at::functionalization::impl::sync(scale_ub.value());
    scale_ub_ =
        at::functionalization::impl::from_functional_tensor(scale_ub.value());
  }

  std::vector<at::Tensor> tmp_output;
  {
    at::AutoDispatchSkipFunctionalize guard;
    tmp_output = vllm_dynamic_per_token_scaled_fp8_quant(out_, input_, scales_,
                                                         scale_ub_);
  }

  at::functionalization::impl::replace_(out, tmp_output[0]);
  at::functionalization::impl::commit_update(out);
  at::functionalization::impl::sync(out);
  at::functionalization::impl::replace_(scales, tmp_output[1]);
  at::functionalization::impl::commit_update(scales);
  at::functionalization::impl::sync(scales);
}

// -------------vllm_dynamic_scaled_fp8_quant-------------------------

// output: out, scales
// input: input
std::vector<at::Tensor> vllm_dynamic_scaled_fp8_quant(
    at::Tensor& out,          // [..., d]
    at::Tensor const& input,  // [..., d]
    at::Tensor& scales) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");

  std::vector<at::Tensor> torch_tensors{input, out, scales};

  auto xla_tensors = bridge::GetXlaTensors(torch_tensors);

  std::string opaque = XlaTensorsToShapeString(xla_tensors);

  TF_VLOG(2) << "opaque: " << opaque;

  auto result_shape = xla::ShapeUtil::MakeTupleShape(
      {xla_tensors[1]->shape().get(), xla_tensors[2]->shape().get()});

  auto result =
      tensor_methods::cuda_custom_call({xla_tensors[0]}, 2, result_shape,
                                       "vllm_dynamic_scaled_fp8_quant", opaque);
  return bridge::AtenFromXlaTensors(result);
}

// buffers[0]: input
// buffers[1]: output
// buffers[2]: scales
void custom_call_vllm_dynamic_scaled_fp8_quant(cudaStream_t stream,
                                               void** buffers,
                                               const char* opaque,
                                               size_t opaque_len) {
  std::string shape_str(opaque, opaque_len);
  auto torch_tensors = ShapeStringToTorchTensors(buffers, shape_str);

  XLA_CHECK(torch_tensors.size() == 3);

  // scales is initialized to 0
  cudaMemsetAsync(buffers[2], 0,
                  torch_tensors[2].numel() * torch_tensors[2].element_size(),
                  stream);

  static const c10::optional<c10::OperatorHandle> op =
      c10::Dispatcher::singleton().findSchema(
          {"_C::dynamic_scaled_fp8_quant", ""});
  XLA_CHECK(op.has_value());

  cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream().stream();
  cudaEvent_t event1;
  cudaEventCreateWithFlags(&event1, cudaEventDisableTiming);
  cudaEvent_t event2;
  cudaEventCreateWithFlags(&event2, cudaEventDisableTiming);
  cudaEventRecord(event1, stream);
  cudaStreamWaitEvent(torch_stream, event1);

  // torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
  callOp(*op, torch_tensors[1], torch_tensors[0], torch_tensors[2]);

  cudaEventRecord(event2, torch_stream);
  cudaStreamWaitEvent(stream, event2);
}

void dynamic_scaled_fp8_quant__functionalization(at::Tensor& out,
                                                 at::Tensor const& input,
                                                 at::Tensor& scales) {
  XLA_CHECK(at::functionalization::impl::isFunctionalTensor(out));
  XLA_CHECK(at::functionalization::impl::isFunctionalTensor(input));
  XLA_CHECK(at::functionalization::impl::isFunctionalTensor(scales));

  at::functionalization::impl::sync(out);
  at::functionalization::impl::sync(input);
  at::functionalization::impl::sync(scales);

  auto out_ = at::functionalization::impl::from_functional_tensor(out);
  auto input_ = at::functionalization::impl::from_functional_tensor(input);
  auto scales_ = at::functionalization::impl::from_functional_tensor(scales);

  std::vector<at::Tensor> tmp_output;
  {
    at::AutoDispatchSkipFunctionalize guard;
    tmp_output = vllm_dynamic_scaled_fp8_quant(out_, input_, scales_);
  }

  at::functionalization::impl::replace_(out, tmp_output[0]);
  at::functionalization::impl::commit_update(out);
  at::functionalization::impl::sync(out);
  at::functionalization::impl::replace_(scales, tmp_output[1]);
  at::functionalization::impl::commit_update(scales);
  at::functionalization::impl::sync(scales);
}

// -------------vllm_static_scaled_fp8_quant-------------------------

// output: out
// input: input, scales
at::Tensor vllm_static_scaled_fp8_quant(at::Tensor& out,
                                        at::Tensor const& input,
                                        at::Tensor& scales) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");

  std::vector<at::Tensor> torch_tensors{input, scales, out};
  auto xla_tensors = bridge::GetXlaTensors(torch_tensors);
  std::string opaque = XlaTensorsToShapeString(xla_tensors);

  TF_VLOG(2) << "opaque: " << opaque;

  auto result = tensor_methods::cuda_custom_call(
      {xla_tensors[0], xla_tensors[1]}, 1, xla_tensors[2]->shape().get(),
      "vllm_static_scaled_fp8_quant", opaque);
  return bridge::AtenFromXlaTensor(result[0]);
}

// buffers[0]: input
// buffers[1]: scales
// buffers[2]: output
void custom_call_vllm_static_scaled_fp8_quant(cudaStream_t stream,
                                              void** buffers,
                                              const char* opaque,
                                              size_t opaque_len) {
  std::string shape_str(opaque, opaque_len);
  auto torch_tensors = ShapeStringToTorchTensors(buffers, shape_str);

  XLA_CHECK(torch_tensors.size() == 3);

  static const c10::optional<c10::OperatorHandle> op =
      c10::Dispatcher::singleton().findSchema(
          {"_C::static_scaled_fp8_quant", ""});
  XLA_CHECK(op.has_value());

  cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream().stream();
  cudaEvent_t event1;
  cudaEventCreateWithFlags(&event1, cudaEventDisableTiming);
  cudaEvent_t event2;
  cudaEventCreateWithFlags(&event2, cudaEventDisableTiming);
  cudaEventRecord(event1, stream);
  cudaStreamWaitEvent(torch_stream, event1);

  // torch.ops._C.static_scaled_fp8_quant(output, input, scale)
  callOp(*op, torch_tensors[2], torch_tensors[0], torch_tensors[1]);

  cudaEventRecord(event2, torch_stream);
  cudaStreamWaitEvent(stream, event2);
}

void static_scaled_fp8_quant__functionalization(at::Tensor& out,
                                                at::Tensor const& input,
                                                at::Tensor const& scales) {
  XLA_CHECK(at::functionalization::impl::isFunctionalTensor(out));
  XLA_CHECK(at::functionalization::impl::isFunctionalTensor(input));
  XLA_CHECK(at::functionalization::impl::isFunctionalTensor(scales));

  at::functionalization::impl::sync(out);
  at::functionalization::impl::sync(input);
  at::functionalization::impl::sync(scales);

  auto out_ = at::functionalization::impl::from_functional_tensor(out);
  auto input_ = at::functionalization::impl::from_functional_tensor(input);
  auto scales_ = at::functionalization::impl::from_functional_tensor(scales);

  at::Tensor tmp_output;
  {
    at::AutoDispatchSkipFunctionalize guard;
    tmp_output = vllm_static_scaled_fp8_quant(out_, input_, scales_);
  }

  at::functionalization::impl::replace_(out, tmp_output);
  at::functionalization::impl::commit_update(out);
  at::functionalization::impl::sync(out);
}

// -------------cutlass_scaled_mm-------------------------

// output: c
// input: a, b, a_scales, b_scales, bias?
at::Tensor vllm_cutlass_scaled_mm(at::Tensor& c, at::Tensor const& a,
                                  at::Tensor const& b,
                                  at::Tensor const& a_scales,
                                  at::Tensor const& b_scales,
                                  c10::optional<at::Tensor> const& bias) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::");
  std::vector<at::Tensor> torch_tensors{a, b, a_scales, b_scales};
  if (bias.has_value()) {
    torch_tensors.push_back(bias.value());
  }
  torch_tensors.push_back(c);

  auto xla_tensors = bridge::GetXlaTensors(torch_tensors);

  std::string opaque = XlaTensorsToShapeString(xla_tensors);

  TF_VLOG(2) << "opaque: " << opaque;

  std::vector<XLATensorPtr> xla_inputs(xla_tensors.begin(),
                                       xla_tensors.end() - 1);
  XLATensorPtr xla_c = xla_tensors.back();

  auto result = tensor_methods::cuda_custom_call(
      xla_inputs, 1, xla_c->shape().get(), "vllm_cutlass_scaled_mm", opaque);
  return bridge::AtenFromXlaTensor(result[0]);
}

// buffers[0]: a
// buffers[1]: b
// buffers[2]: a_scales
// buffers[3]: b_scales
// buffers[4]: bias?
// buffers[5]: output
void custom_call_vllm_cutlass_scaled_mm(cudaStream_t stream, void** buffers,
                                        const char* opaque, size_t opaque_len) {
  std::string shape_str(opaque, opaque_len);
  auto torch_tensors = ShapeStringToTorchTensors(buffers, shape_str);

  XLA_CHECK(torch_tensors.size() == 5 || torch_tensors.size() == 6);

  c10::optional<torch::Tensor> bias;
  if (torch_tensors.size() == 6) {
    bias = torch_tensors[4];
  }

  static const c10::optional<c10::OperatorHandle> op =
      c10::Dispatcher::singleton().findSchema({"_C::cutlass_scaled_mm", ""});
  XLA_CHECK(op.has_value());

  cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream().stream();
  cudaEvent_t event1;
  cudaEventCreateWithFlags(&event1, cudaEventDisableTiming);
  cudaEvent_t event2;
  cudaEventCreateWithFlags(&event2, cudaEventDisableTiming);
  cudaEventRecord(event1, stream);
  cudaStreamWaitEvent(torch_stream, event1);

  // torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)
  callOp(*op, torch_tensors.back(), torch_tensors[0], torch_tensors[1],
         torch_tensors[2], torch_tensors[3], bias);

  cudaEventRecord(event2, torch_stream);
  cudaStreamWaitEvent(stream, event2);
}

void cutlass_scaled_mm__functionalization(
    at::Tensor& c, at::Tensor const& a, at::Tensor const& b,
    at::Tensor const& a_scales, at::Tensor const& b_scales,
    c10::optional<at::Tensor> const& bias) {
  std::vector<at::Tensor> torch_tensors{c, a, b, a_scales, b_scales};
  if (bias.has_value()) {
    torch_tensors.push_back(bias.value());
  }

  std::vector<at::Tensor> unwrap_tensors;
  for (auto& torch_tensor : torch_tensors) {
    XLA_CHECK(at::functionalization::impl::isFunctionalTensor(torch_tensor));
    at::functionalization::impl::sync(torch_tensor);
    unwrap_tensors.push_back(
        at::functionalization::impl::from_functional_tensor(torch_tensor));
  }

  c10::optional<at::Tensor> unwrap_bias;
  if (bias.has_value()) {
    unwrap_bias = unwrap_tensors[5];
  }

  at::Tensor tmp_output;
  {
    at::AutoDispatchSkipFunctionalize guard;
    tmp_output = vllm_cutlass_scaled_mm(unwrap_tensors[0], unwrap_tensors[1],
                                        unwrap_tensors[2], unwrap_tensors[3],
                                        unwrap_tensors[4], unwrap_bias);
  }

  at::functionalization::impl::replace_(c, tmp_output);
  at::functionalization::impl::commit_update(c);
  at::functionalization::impl::sync(c);
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "vllm_dynamic_per_token_scaled_fp8_quant",
    custom_call_vllm_dynamic_per_token_scaled_fp8_quant, "CUDA");

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "vllm_dynamic_scaled_fp8_quant", custom_call_vllm_dynamic_scaled_fp8_quant,
    "CUDA");

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(
    "vllm_static_scaled_fp8_quant", custom_call_vllm_static_scaled_fp8_quant,
    "CUDA");

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("vllm_cutlass_scaled_mm",
                                         custom_call_vllm_cutlass_scaled_mm,
                                         "CUDA");

TORCH_LIBRARY_IMPL(_C, Functionalize, m) {
  m.impl("dynamic_per_token_scaled_fp8_quant",
         dynamic_per_token_scaled_fp8_quant__functionalization);
  m.impl("dynamic_scaled_fp8_quant",
         dynamic_scaled_fp8_quant__functionalization);
  m.impl("static_scaled_fp8_quant", static_scaled_fp8_quant__functionalization);
  m.impl("cutlass_scaled_mm", cutlass_scaled_mm__functionalization);
}

}  // namespace torch_xla
