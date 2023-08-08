#ifndef XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_FORWARD_H_
#define XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_FORWARD_H_

#include "torch_xla/csrc/flash_attention_utils.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class FlashAttentionForward : public XlaNode {
 public:
  FlashAttentionForward(const torch::lazy::Value& q,
                        const torch::lazy::Value& k,
                        const torch::lazy::Value& v,
                        const torch::lazy::Value& cu_seqlens_q,
                        const torch::lazy::Value& cu_seqlens_k,
                        const FlashAttentionForwardParams& params);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  FlashAttentionForwardParams params_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_FLASH_ATTENTION_FORWARD_H_