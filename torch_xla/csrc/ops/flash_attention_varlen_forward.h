#ifndef XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_VARLEN_FORWARD_H_
#define XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_VARLEN_FORWARD_H_

#include "torch_xla/csrc/flash_attention_utils.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class FlashAttentionVarlenForward : public XlaNode {
 public:
  FlashAttentionVarlenForward(const torch::lazy::Value& q,
                              const torch::lazy::Value& k,
                              const torch::lazy::Value& v,
                              const torch::lazy::Value& attention_mask,
                              const FlashAttentionForwardParams& params,
                              const std::string& params_str);

  FlashAttentionVarlenForward(const torch::lazy::Value& q,
                              const torch::lazy::Value& k,
                              const torch::lazy::Value& v,
                              const torch::lazy::Value& attention_mask,
                              const torch::lazy::Value& alibi_slopes,
                              const FlashAttentionForwardParams& params,
                              const std::string& params_str);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  FlashAttentionForwardParams params_;
  const std::string params_str_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_VARLEN_FORWARD_H_