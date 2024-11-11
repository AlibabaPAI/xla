#include "torch_xla/csrc/ops/view_symint.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace {

std::vector<torch::lazy::Value> GetValues(
    const torch::lazy::Value& input,
    const std::vector<torch::lazy::NodePtr>& dimensions) {
  std::vector<torch::lazy::Value> values;
  values.reserve(dimensions.size() + 1);
  values.push_back(input);
  for (torch::lazy::NodePtr dim : dimensions) {
    if (dim) {
      // Dimension Node only exist for symbolic dimension.
      values.push_back(torch::lazy::Value(dim, 0));
    }
  }
  return values;
}

}  // namespace

ViewSymIntOp::ViewSymIntOp(const torch::lazy::Value& input,
                           const SymIntElements& size_elements,
                           xla::Shape output_shape)
    : XlaNode(
          torch::lazy::OpKind(at::aten::view),
          GetValues(input, size_elements.GetSizeNodes()), output_shape,
          /*num_outputs=*/1,
          torch::lazy::MHash(
              torch::lazy::ToVector<int64_t>(output_shape.dimensions()),
              torch::lazy::ToVector<bool>(output_shape.dynamic_dimensions()))),
      upper_bounds_(torch::lazy::ToVector<int64_t>(output_shape.dimensions())),
      dynamic_dims_(
          torch::lazy::ToVector<bool>(output_shape.dynamic_dimensions())) {}

XlaOpVector ViewSymIntOp::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  std::vector<xla::XlaOp> size_ops;
  for (int i = 1; i < operands().size(); i++) {
    size_ops.push_back(loctx->GetOutputOp(operand(i)));
  }
  xla::XlaOp output =
      BuildViewSymInt(input, size_ops, upper_bounds_, dynamic_dims_);
  return ReturnOp(output, loctx);
}

std::string ViewSymIntOp::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", size=(" << absl::StrJoin(upper_bounds_, ", ")
     << ")"
     << ", dynamic_dims=(" << absl::StrJoin(dynamic_dims_, ", ") << ")";
  return ss.str();
}

}  // namespace torch_xla
