#ifndef XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_BACKWARD_H_
#define XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_BACKWARD_H_

#include "torch_xla/csrc/flash_attention_utils.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class FlashAttentionBackward : public XlaNode {
 public:
  FlashAttentionBackward(const torch::lazy::Value& dout,
                         const torch::lazy::Value& q,
                         const torch::lazy::Value& k,
                         const torch::lazy::Value& v,
                         const torch::lazy::Value& out,
                         const torch::lazy::Value& softmax_lse,
                         const torch::lazy::Value& cu_seqlens_q,
                         const torch::lazy::Value& cu_seqlens_k,
                         const torch::lazy::Value& rng_state,
                         const FlashAttentionBackwardParams& params);

  FlashAttentionBackward(const torch::lazy::Value& dout,
                         const torch::lazy::Value& q,
                         const torch::lazy::Value& k,
                         const torch::lazy::Value& v,
                         const torch::lazy::Value& out,
                         const torch::lazy::Value& softmax_lse,
                         const torch::lazy::Value& cu_seqlens_q,
                         const torch::lazy::Value& cu_seqlens_k,
                         const torch::lazy::Value& rng_state,
                         const torch::lazy::Value& alibi_slopes,
                         const FlashAttentionBackwardParams& params);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  FlashAttentionBackwardParams params_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_BACKWARD_H_