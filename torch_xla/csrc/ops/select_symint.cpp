#include "torch_xla/csrc/ops/select_symint.h"

#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/shape_helper.h"

namespace torch_xla {

SelectSymInt::SelectSymInt(const torch::lazy::Value& input, int64_t dim,
                           const torch::lazy::Value& start,
                           const torch::lazy::Value& end,
                           const torch::lazy::Value& stride,
                           xla::Shape output_shape)
    : XlaNode(xla_select, {input, start, end, stride}, output_shape,
              /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

XlaOpVector SelectSymInt::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp start = loctx->GetOutputOp(operand(1));
  xla::XlaOp end = loctx->GetOutputOp(operand(2));
  xla::XlaOp stride = loctx->GetOutputOp(operand(3));
  xla::XlaOp output =
      BuildSelectSymInt(input, dim_, start, end, stride, xla_shape());
  return ReturnOp(output, loctx);
}

std::string SelectSymInt::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

xla::Shape SelectSymInt::MakeSelectShape(const xla::Shape& shape, int64_t dim,
                                         int64_t start, int64_t end,
                                         int64_t stride) {
  int64_t effective_stride = GetStride(start, end, stride);
  xla::Shape select_shape(shape);
  select_shape.set_dimensions(
      dim, (end - start + effective_stride - 1) / effective_stride);
  select_shape.set_dynamic_dimension(dim, true);
  return select_shape;
}

int64_t SelectSymInt::GetStride(int64_t start, int64_t end, int64_t stride) {
  if (stride == 0) {
    XLA_CHECK_EQ(start, end);
    stride = 1;
  }
  return stride;
}

}  // namespace torch_xla
