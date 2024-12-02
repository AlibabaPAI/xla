#ifndef XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_VARLEN_POSITION_IDS_BACKWARD_H_
#define XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_VARLEN_POSITION_IDS_BACKWARD_H_

#include "torch_xla/csrc/flash_attention_utils.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class FlashAttentionVarlenPositionIdsBackward : public XlaNode {
 public:
  FlashAttentionVarlenPositionIdsBackward(
      const torch::lazy::Value& dout, const torch::lazy::Value& q,
      const torch::lazy::Value& k, const torch::lazy::Value& v,
      const torch::lazy::Value& out, const torch::lazy::Value& softmax_lse,
      const torch::lazy::Value& cu_seqlens_q,
      const torch::lazy::Value& cu_seqlens_k,
      const torch::lazy::Value& rng_state, const std::string params);

  FlashAttentionVarlenPositionIdsBackward(
      const torch::lazy::Value& dout, const torch::lazy::Value& q,
      const torch::lazy::Value& k, const torch::lazy::Value& v,
      const torch::lazy::Value& out, const torch::lazy::Value& softmax_lse,
      const torch::lazy::Value& cu_seqlens_q,
      const torch::lazy::Value& cu_seqlens_k,
      const torch::lazy::Value& rng_state,
      const torch::lazy::Value& alibi_slopes, const std::string params);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  const std::string params_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_VARLEN_POSITION_IDS_BACKWARD_H_