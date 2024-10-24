#ifndef XLA_TORCH_XLA_CSRC_TOKEN_HANDLER_H_
#define XLA_TORCH_XLA_CSRC_TOKEN_HANDLER_H_

#include "xla/client/xla_builder.h"

namespace torch_xla {

class TokenHandler {
 public:
  explicit TokenHandler(xla::XlaOp token) : token_(token) {}

  xla::XlaOp GetInput(xla::XlaOp input, const xla::Shape* input_shape);
  xla::XlaOp GetNewOutput(xla::XlaOp result);
  xla::XlaOp GetNewToken();

 private:
  xla::XlaOp token_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_TOKEN_HANDLER_H_