#ifndef XLA_TORCH_XLA_CSRC_OPS_SELECT_SYMINT_H_
#define XLA_TORCH_XLA_CSRC_OPS_SELECT_SYMINT_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class SelectSymInt : public XlaNode {
 public:
  SelectSymInt(const torch::lazy::Value& input,
                             int64_t dim,
                             const torch::lazy::Value& start,
                             const torch::lazy::Value& end,
                             const torch::lazy::Value& stride,
                             xla::Shape output_shape);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t dim() const { return dim_; }

  static xla::Shape MakeSelectShape(const xla::Shape& shape, int64_t dim,
                                    int64_t start, int64_t end, int64_t stride);

  static int64_t GetStride(int64_t start, int64_t end, int64_t stride);

 private:
  int64_t dim_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_SELECT_SYMINT_H_