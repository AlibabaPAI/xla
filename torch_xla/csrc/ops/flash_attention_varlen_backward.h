#ifndef XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_VARLEN_BACKWARD_H_
#define XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_VARLEN_BACKWARD_H_

#include "torch_xla/csrc/flash_attention_utils.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class FlashAttentionVarlenBackward : public XlaNode {
 public:
  FlashAttentionVarlenBackward(const torch::lazy::Value& dout,
                               const torch::lazy::Value& q,
                               const torch::lazy::Value& k,
                               const torch::lazy::Value& v,
                               const torch::lazy::Value& out,
                               const torch::lazy::Value& softmax_lse,
                               const torch::lazy::Value& cu_seqlens_q,
                               const torch::lazy::Value& cu_seqlens_k,
                               const torch::lazy::Value& rng_state,
                               const FlashAttentionBackwardParams& params,
                               const std::string& params_str);

  FlashAttentionVarlenBackward(const torch::lazy::Value& dout,
                               const torch::lazy::Value& q,
                               const torch::lazy::Value& k,
                               const torch::lazy::Value& v,
                               const torch::lazy::Value& out,
                               const torch::lazy::Value& softmax_lse,
                               const torch::lazy::Value& cu_seqlens_q,
                               const torch::lazy::Value& cu_seqlens_k,
                               const torch::lazy::Value& rng_state,
                               const torch::lazy::Value& alibi_slopes,
                               const FlashAttentionBackwardParams& params,
                               const std::string& params_str);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  FlashAttentionBackwardParams params_;
  const std::string params_str_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_VARLEN_BACKWARD_H_