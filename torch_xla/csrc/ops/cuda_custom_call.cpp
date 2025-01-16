#include "torch_xla/csrc/ops/cuda_custom_call.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

CudaCustomCall::CudaCustomCall(torch::lazy::OpList inputs, int num_outputs,
                               xla::Shape output_shape,
                               const std::string& call_target_name,
                               const std::string& opaque)
    : XlaNode(xla_cuda_custom_call, inputs, std::move(output_shape),
              num_outputs, torch::lazy::MHash(call_target_name + opaque)),
      call_target_name_(call_target_name),
      opaque_(opaque) {}

torch::lazy::NodePtr CudaCustomCall::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<CudaCustomCall>(
      operands, num_outputs(), xla_shape(), call_target_name_, opaque_);
}

XlaOpVector CudaCustomCall::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  inputs.reserve(operands().size());
  for (auto& operand : operands()) {
    inputs.push_back(loctx->GetOutputOp(operand));
  }
  std::vector<xla::XlaOp> output = BuildCudaCustomCall(
      inputs, num_outputs(), xla_shape(), call_target_name_, opaque_);
  if (num_outputs() == 1) {
    return ReturnOp(output[0], loctx);
  }
  return ReturnOps(output, loctx);
}

std::string CudaCustomCall::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", call_target_name=" << call_target_name_
     << ", opaque=" << opaque_;
  return ss.str();
}

}  // namespace torch_xla
