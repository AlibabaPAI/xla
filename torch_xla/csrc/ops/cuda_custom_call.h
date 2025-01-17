#ifndef XLA_TORCH_XLA_CSRC_OPS_CUDA_CUSTOM_CALL_H_
#define XLA_TORCH_XLA_CSRC_OPS_CUDA_CUSTOM_CALL_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class CudaCustomCall : public XlaNode {
 public:
  CudaCustomCall(torch::lazy::OpList inputs, int num_outputs,
                 xla::Shape output_shape, const std::string& call_target_name,
                 const std::string& opaque);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

 private:
  std::string call_target_name_;
  std::string opaque_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_CUDA_CUSTOM_CALL_H_
