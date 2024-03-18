#include "torch_xla/csrc/ops/upsample_bilinear2d_backward.h"

#include <ATen/Operators.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAStream.h>

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/dtype.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/resize_ops.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "tsl/platform/human_readable_json.h"
#include "xla/service/custom_call_target_registry.h"

namespace {

void upsample_bilinear_backward(cudaStream_t stream, void** buffers,
                                const char* opaque, size_t opaque_len) {
  // Parse config information from opaque
  std::string backend_config(opaque, opaque_len);
  std::vector<std::string> config_list = absl::StrSplit(backend_config, "|");

  // Parse input_shape and output_shape
  xla::ShapeProto input_shape_proto;
  xla::ShapeProto output_shape_proto;
  TF_CHECK_OK(
      tsl::HumanReadableJsonToProto(config_list[0], &input_shape_proto));
  TF_CHECK_OK(
      tsl::HumanReadableJsonToProto(config_list[1], &output_shape_proto));
  xla::Shape input_shape(input_shape_proto);
  xla::Shape output_shape(output_shape_proto);

  // Parse align_corners
  bool align_corners;
  absl::SimpleAtob(config_list[2], &align_corners);

  // Parse scales_h and scales_w
  c10::optional<double> scales_h;
  c10::optional<double> scales_w;
  if (config_list[3] != "null") {
    double scales_h_val;
    absl::SimpleAtod(config_list[3], &scales_h_val);
    scales_h = scales_h_val;
  }
  if (config_list[4] != "null") {
    double scales_w_val;
    absl::SimpleAtod(config_list[4], &scales_w_val);
    scales_w = scales_w_val;
  }

  std::vector<std::string> output_size_str_vec =
      absl::StrSplit(config_list[5], ",");
  std::vector<int64_t> output_size(2);
  absl::SimpleAtoi(output_size_str_vec[0], &output_size[0]);
  absl::SimpleAtoi(output_size_str_vec[1], &output_size[1]);
  c10::SymIntArrayRef output_size_symint =
      c10::fromIntArrayRefSlow(output_size);

  std::vector<std::string> input_size_str_vec =
      absl::StrSplit(config_list[6], ",");
  std::vector<int64_t> input_size(4);
  absl::SimpleAtoi(input_size_str_vec[0], &input_size[0]);
  absl::SimpleAtoi(input_size_str_vec[1], &input_size[1]);
  absl::SimpleAtoi(input_size_str_vec[2], &input_size[2]);
  absl::SimpleAtoi(input_size_str_vec[3], &input_size[3]);
  c10::SymIntArrayRef input_size_symint = c10::fromIntArrayRefSlow(input_size);

  // Generate at::Tensor from xla buffer
  auto scalar_type =
      torch_xla::TorchTypeFromXlaType(input_shape.element_type());
  auto opts = at::TensorOptions().dtype(scalar_type).device(at::kCUDA);
  at::Tensor input = at::from_blob(
      buffers[0],
      {input_shape.dimensions().begin(), input_shape.dimensions().end()}, opts);
  at::Tensor output = at::from_blob(
      buffers[1],
      {output_shape.dimensions().begin(), output_shape.dimensions().end()},
      opts);

  // Run the upsample_bilinear2d_backward aten op
  cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream().stream();
  cudaEvent_t event1;
  cudaEventCreateWithFlags(&event1, cudaEventDisableTiming);
  cudaEvent_t event2;
  cudaEventCreateWithFlags(&event2, cudaEventDisableTiming);
  cudaEventRecord(event1, stream);
  cudaStreamWaitEvent(torch_stream, event1);
  ATEN_OP2(upsample_bilinear2d_backward, grad_input)::redispatch(
      c10::DispatchKeySet(c10::DispatchKey::CUDA), input, output_size_symint,
      input_size_symint, align_corners, scales_h, scales_w, output);
  cudaEventRecord(event2, torch_stream);
  cudaStreamWaitEvent(stream, event2);
}

XLA_REGISTER_CUSTOM_CALL_TARGET(upsample_bilinear_backward, "CUDA");

}  // namespace

namespace torch_xla {

UpsampleBilinearBackward::UpsampleBilinearBackward(
    const torch::lazy::Value& input, std::vector<int64_t> output_size,
    std::vector<int64_t> input_size, bool align_corners,
    c10::optional<double> scales_h, c10::optional<double> scales_w)
    : XlaNode(torch::lazy::OpKind(at::aten::upsample_bilinear2d_backward),
              {input},
              [&]() {
                return resize::GetBackwardOutputShape2d(GetXlaShape(input),
                                                        input_size);
              },
              /*num_outputs=*/1,
              torch::lazy::MHash(output_size, input_size, align_corners)),
      output_size_(std::move(output_size)),
      input_size_(std::move(input_size)),
      align_corners_(align_corners),
      scales_h_(scales_h),
      scales_w_(scales_w) {}

torch::lazy::NodePtr UpsampleBilinearBackward::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<UpsampleBilinearBackward>(
      operands.at(0), output_size_, input_size_, align_corners_, scales_h_,
      scales_w_);
}

XlaOpVector UpsampleBilinearBackward::Lower(LoweringContext* loctx) const {
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(bridge::GetCurrentDevice().type());
  if (hw_type == XlaDeviceType::TPU || hw_type == XlaDeviceType::NEURON) {
    xla::XlaOp input = loctx->GetOutputOp(operand(0));
    xla::XlaOp output = resize::LowerBackward2d(
        "ResizeBilinearGrad", input, xla_shape(), align_corners_,
        /*half_pixel_centers=*/!align_corners_);
    return ReturnOp(output, loctx);
  }
  TF_VLOG(2) << "Lowering UpsampleBilinearBackward";
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  std::string input_shape_str;
  std::string output_shape_str;
  std::string align_corners_str = absl::StrCat(align_corners_);
  std::string scales_h_str =
      scales_w_.has_value() ? absl::StrCat(*scales_h_) : "null";
  std::string scales_w_str =
      scales_w_.has_value() ? absl::StrCat(*scales_w_) : "null";
  std::string output_size_str = absl::StrJoin(output_size_, ", ");
  std::string input_size_str = absl::StrJoin(input_size_, ", ");
  TF_CHECK_OK(tsl::ProtoToHumanReadableJson(
      loctx->builder()->GetShape(input).value().ToProto(), &input_shape_str,
      /*ignore_accuracy_loss=*/true));
  TF_CHECK_OK(tsl::ProtoToHumanReadableJson(xla_shape().ToProto(),
                                            &output_shape_str,
                                            /*ignore_accuracy_loss=*/true));

  auto opaque = absl::StrJoin(
      {input_shape_str, output_shape_str, align_corners_str, scales_h_str,
       scales_w_str, output_size_str, input_size_str},
      "|");
  TF_VLOG(2) << "opaque: " << opaque;
  xla::XlaOp output =
      xla::CustomCall(input.builder(), "upsample_bilinear_backward", {input},
                      xla_shape(), opaque);

  return ReturnOp(output, loctx);
}

std::string UpsampleBilinearBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << "), input_size=("
     << absl::StrJoin(input_size_, ", ")
     << "), align_corners=" << align_corners_;
  return ss.str();
}

}  // namespace torch_xla
