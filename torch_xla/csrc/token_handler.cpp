#include "torch_xla/csrc/token_handler.h"

#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/shape_helper.h"
#include "xla/client/lib/constants.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace {

xla::XlaOp SliceOneToken(xla::XlaOp input) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  int64_t input_rank = input_shape.rank();
  if (input_rank > 0) {
    xla::GatherDimensionNumbers dim_numbers;
    for (int64_t i = 0; i < input_rank; ++i) {
      dim_numbers.add_collapsed_slice_dims(i);
      dim_numbers.add_start_index_map(i);
    }
    dim_numbers.set_index_vector_dim(0);

    std::vector<int64_t> slice_sizes(input_rank, 1);
    xla::XlaOp indices = xla::Zeros(
        input.builder(),
        xla::ShapeUtil::MakeShape(xla::PrimitiveType::S32, {input_rank}));
    input = xla::Gather(input, indices, dim_numbers, slice_sizes);
  }
  return input;
}

}  // namespace

xla::XlaOp TokenHandler::GetInput(xla::XlaOp input,
                                  const xla::Shape* input_shape) {
  static bool disable_numeric_token =
      runtime::sys_util::GetEnvBool("DISABLE_NUMERIC_CC_TOKEN", false);
  if (disable_numeric_token) {
    return input;
  }

  if (input_shape == nullptr) {
    input_shape = &ShapeHelper::ShapeOfXlaOp(input);
  }
  // Token is always a numeric zero, so adding to input does not change input.
  return input + MaybeConvertTo(token_, input_shape->element_type());
}

xla::XlaOp TokenHandler::GetNewOutput(xla::XlaOp result) {
  static bool disable_numeric_token =
      runtime::sys_util::GetEnvBool("DISABLE_NUMERIC_CC_TOKEN", false);
  if (disable_numeric_token) {
    return result;
  }

  xla::XlaOp tuple_input = xla::Tuple(result.builder(), {result, token_});
  xla::XlaOp tuple_output = xla::OptimizationBarrier(tuple_input);

  token_ = xla::GetTupleElement(tuple_output, 1);

  return xla::GetTupleElement(tuple_output, 0);
}

xla::XlaOp TokenHandler::GetNewToken() { return token_; }

}  // namespace torch_xla
