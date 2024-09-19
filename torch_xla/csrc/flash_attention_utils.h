#ifndef XLA_TORCH_XLA_CSRC_FLASH_ATTENTION_UTILS_H
#define XLA_TORCH_XLA_CSRC_FLASH_ATTENTION_UTILS_H

#include <torch/extension.h>

namespace torch_xla {

struct FlashAttentionBaseParams {
  using index_t = uint32_t;

  // The stride between rows of the Q, K and V matrices.
  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;

  // The number of heads.
  int h, h_k;
  // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k
  // could be different from nheads (query).
  int h_h_k_ratio;  // precompute h / h_k,

  virtual std::string ToString() const;
  virtual void FromString(const std::string& str);
};

struct FlashAttentionForwardParams : public FlashAttentionBaseParams {
  // The stride between rows of O.
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  // The dimensions.
  // \seqlen_q, \seqlen_k are the padding seqlen
  int b, seqlen_q, seqlen_k, d, d_rounded;

  // The scaling factors for the kernel.
  float scale_softmax;
  float scale_softmax_log2;

  // The dropout probability (probability of keeping an activation).
  float p_dropout;
  uint8_t p_dropout_in_uint8_t;

  // Scale factor of 1 / (1 - p_dropout).
  float rp_dropout;
  float scale_softmax_rp_dropout;

  bool is_bf16;
  bool is_causal;
  int window_size_left;
  int window_size_right;
  int alibi_slopes_batch_stride;
  bool enable_alibi_slopes;
  bool is_seqlens_k_cumulative;
  int num_splits;

  virtual std::string ToString() const override;
  virtual void FromString(const std::string& str) override;
};

struct FlashAttentionBackwardParams : public FlashAttentionForwardParams {
  index_t do_batch_stride;
  index_t do_row_stride;
  index_t do_head_stride;
  index_t dq_batch_stride;
  index_t dk_batch_stride;
  index_t dv_batch_stride;
  index_t dq_row_stride;
  index_t dk_row_stride;
  index_t dv_row_stride;
  index_t dq_head_stride;
  index_t dk_head_stride;
  index_t dv_head_stride;

  bool deterministic;

  std::string ToString() const override;
  void FromString(const std::string& str) override;
};

void set_forward_params(
    FlashAttentionForwardParams& params, const size_t b, const size_t seqlen_q,
    const size_t seqlen_k, const size_t h, const size_t h_k, const size_t d,
    const size_t d_rounded, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, void* attention_mask, float p_dropout,
    float softmax_scale, bool is_causal, int window_size_left,
    int window_size_right, int alibi_slopes_batch_stride,
    bool enable_alibi_slopes,
    bool seqlenq_ngroups_swapped =
        false /*TODO(wenting.swt): support max_seqlen_q==1*/);

void set_backward_params(FlashAttentionBackwardParams& params, const size_t b,
                         const size_t seqlen_q, const size_t seqlen_k,
                         const size_t h, const size_t h_k, const size_t d,
                         const size_t d_rounded, const at::Tensor& q,
                         const at::Tensor& k, const at::Tensor& v,
                         const at::Tensor& dout, float p_dropout,
                         float softmax_scale, bool is_causal,
                         int window_size_left, int window_size_right,
                         bool deterministic, int alibi_slopes_batch_stride,
                         bool enable_alibi_slopes);

FlashAttentionForwardParams get_flash_attention_forward_params(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    c10::optional<at::Tensor>& attention_mask,
    c10::optional<at::Tensor>& alibi_slopes_, const float p_dropout,
    const float softmax_scale, const bool zero_tensors, const bool is_causal,
    int window_size_left, int window_size_right, const bool return_softmax);

FlashAttentionBackwardParams get_flash_attention_backward_params(
    const at::Tensor& dout, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& out, const at::Tensor& softmax_lse,
    c10::optional<at::Tensor>& cu_seqlens_q,
    c10::optional<at::Tensor>& cu_seqlens_k,
    c10::optional<at::Tensor>& alibi_slopes_, const float p_dropout,
    const float softmax_scale, const bool zero_tensors, const bool is_causal,
    int window_size_left, int window_size_right, const bool deterministic);

at::Tensor cu_seqlens_to_indices(const at::Tensor& cu_seqlens, int batch_size,
                                 int seqlen, torch::Dtype scalar_type,
                                 int& max_seqlen_in_batch, int& total);

at::Tensor mask_to_indices(const at::Tensor& attention_mask,
                           int& max_seqlen_in_batch, int& total,
                           at::Tensor& cu_seqlen);

at::Tensor index_first_axis(const at::Tensor& input, const at::Tensor& indices);

}  // namespace torch_xla
#endif  // XLA_TORCH_XLA_CSRC_FLASH_ATTENTION_UTILS_H