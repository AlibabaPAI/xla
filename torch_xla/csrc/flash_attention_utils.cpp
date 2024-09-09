#include "torch_xla/csrc/flash_attention_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/tf_logging.h"

namespace torch_xla {

#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")

std::string FlashAttentionBaseParams::ToString() const {
  XLA_ERROR() << "ToString not implemented";
}

void FlashAttentionBaseParams::FromString(const std::string& str) {
  XLA_ERROR() << "FromString not implemented";
}

std::string FlashAttentionForwardParams::ToString() const {
  std::string result = "";
  absl::StrAppend(&result, absl::StrCat(this->q_batch_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->k_batch_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->v_batch_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->q_row_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->k_row_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->v_row_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->q_head_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->k_head_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->v_head_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->h), "|");
  absl::StrAppend(&result, absl::StrCat(this->h_k), "|");
  absl::StrAppend(&result, absl::StrCat(this->h_h_k_ratio), "|");
  absl::StrAppend(&result, absl::StrCat(this->o_batch_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->o_row_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->o_head_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->b), "|");
  absl::StrAppend(&result, absl::StrCat(this->seqlen_q), "|");
  absl::StrAppend(&result, absl::StrCat(this->seqlen_k), "|");
  absl::StrAppend(&result, absl::StrCat(this->d), "|");
  absl::StrAppend(&result, absl::StrCat(this->d_rounded), "|");
  absl::StrAppend(&result, absl::StrCat(this->scale_softmax), "|");
  absl::StrAppend(&result, absl::StrCat(this->scale_softmax_log2), "|");
  absl::StrAppend(&result, absl::StrCat(this->p_dropout), "|");
  absl::StrAppend(&result, absl::StrCat(uint32_t(this->p_dropout_in_uint8_t)),
                  "|");
  absl::StrAppend(&result, absl::StrCat(this->rp_dropout), "|");
  absl::StrAppend(&result, absl::StrCat(this->scale_softmax_rp_dropout), "|");
  absl::StrAppend(&result, absl::StrCat(this->is_bf16), "|");
  absl::StrAppend(&result, absl::StrCat(this->is_causal), "|");
  absl::StrAppend(&result, absl::StrCat(this->window_size_left), "|");
  absl::StrAppend(&result, absl::StrCat(this->window_size_right), "|");
  absl::StrAppend(&result, absl::StrCat(this->alibi_slopes_batch_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->is_seqlens_k_cumulative), "|");
  absl::StrAppend(&result, absl::StrCat(this->num_splits), "|");
  absl::StrAppend(&result, absl::StrCat(this->enable_alibi_slopes));

  return result;
}

void FlashAttentionForwardParams::FromString(const std::string& str) {
  std::vector<std::string> params_list = absl::StrSplit(str, "|");
  TORCH_CHECK(params_list.size() >= 34);  // at least 38 variables
  absl::SimpleAtoi(params_list[0], &this->q_batch_stride);
  absl::SimpleAtoi(params_list[1], &this->k_batch_stride);
  absl::SimpleAtoi(params_list[2], &this->v_batch_stride);
  absl::SimpleAtoi(params_list[3], &this->q_row_stride);
  absl::SimpleAtoi(params_list[4], &this->k_row_stride);
  absl::SimpleAtoi(params_list[5], &this->v_row_stride);
  absl::SimpleAtoi(params_list[6], &this->q_head_stride);
  absl::SimpleAtoi(params_list[7], &this->k_head_stride);
  absl::SimpleAtoi(params_list[8], &this->v_head_stride);
  absl::SimpleAtoi(params_list[9], &this->h);
  absl::SimpleAtoi(params_list[10], &this->h_k);
  absl::SimpleAtoi(params_list[11], &this->h_h_k_ratio);
  absl::SimpleAtoi(params_list[12], &this->o_batch_stride);
  absl::SimpleAtoi(params_list[13], &this->o_row_stride);
  absl::SimpleAtoi(params_list[14], &this->o_head_stride);
  absl::SimpleAtoi(params_list[15], &this->b);
  absl::SimpleAtoi(params_list[16], &this->seqlen_q);
  absl::SimpleAtoi(params_list[17], &this->seqlen_k);
  absl::SimpleAtoi(params_list[18], &this->d);
  absl::SimpleAtoi(params_list[19], &this->d_rounded);
  absl::SimpleAtof(params_list[20], &this->scale_softmax);
  absl::SimpleAtof(params_list[21], &this->scale_softmax_log2);
  absl::SimpleAtof(params_list[22], &this->p_dropout);
  uint32_t tmp;
  absl::SimpleAtoi(params_list[23], &tmp);
  this->p_dropout_in_uint8_t = uint8_t(tmp);
  absl::SimpleAtof(params_list[24], &this->rp_dropout);
  absl::SimpleAtof(params_list[25], &this->scale_softmax_rp_dropout);
  absl::SimpleAtob(params_list[26], &this->is_bf16);
  absl::SimpleAtob(params_list[27], &this->is_causal);
  absl::SimpleAtoi(params_list[28], &this->window_size_left);
  absl::SimpleAtoi(params_list[29], &this->window_size_right);
  absl::SimpleAtoi(params_list[30], &this->alibi_slopes_batch_stride);
  absl::SimpleAtob(params_list[31], &this->is_seqlens_k_cumulative);
  absl::SimpleAtoi(params_list[32], &this->num_splits);
  absl::SimpleAtob(params_list[33], &this->enable_alibi_slopes);
}

std::string FlashAttentionBackwardParams::ToString() const {
  std::string result = FlashAttentionForwardParams::ToString();
  absl::StrAppend(&result, "|");
  absl::StrAppend(&result, absl::StrCat(this->do_batch_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->do_row_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->do_head_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->dq_batch_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->dk_batch_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->dv_batch_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->dq_row_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->dk_row_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->dv_row_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->dq_head_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->dk_head_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->dv_head_stride), "|");
  absl::StrAppend(&result, absl::StrCat(this->deterministic));
  return result;
}

void FlashAttentionBackwardParams::FromString(const std::string& str) {
  FlashAttentionForwardParams::FromString(str);
  std::vector<std::string> params_list = absl::StrSplit(str, "|");
  TORCH_CHECK(params_list.size() == 47);
  const int offset = 34;  // FlashAttentionForwardParams has 38 variables
  absl::SimpleAtoi(params_list[offset + 0], &this->do_batch_stride);
  absl::SimpleAtoi(params_list[offset + 1], &this->do_row_stride);
  absl::SimpleAtoi(params_list[offset + 2], &this->do_head_stride);
  absl::SimpleAtoi(params_list[offset + 3], &this->dq_batch_stride);
  absl::SimpleAtoi(params_list[offset + 4], &this->dk_batch_stride);
  absl::SimpleAtoi(params_list[offset + 5], &this->dv_batch_stride);
  absl::SimpleAtoi(params_list[offset + 6], &this->dq_row_stride);
  absl::SimpleAtoi(params_list[offset + 7], &this->dk_row_stride);
  absl::SimpleAtoi(params_list[offset + 8], &this->dv_row_stride);
  absl::SimpleAtoi(params_list[offset + 9], &this->dq_head_stride);
  absl::SimpleAtoi(params_list[offset + 10], &this->dk_head_stride);
  absl::SimpleAtoi(params_list[offset + 11], &this->dv_head_stride);
  absl::SimpleAtob(params_list[offset + 12], &this->deterministic);
}

void set_forward_params(
    FlashAttentionForwardParams& params, const size_t b, const size_t seqlen_q,
    const size_t seqlen_k, const size_t h, const size_t h_k,
    const size_t d, const size_t d_rounded, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, float p_dropout,
    float softmax_scale, bool is_causal, int window_size_left,
    int window_size_right, int alibi_slopes_batch_stride,
    bool enable_alibi_slopes, bool seqlenq_ngroups_swapped) {
  // Reset the parameters
  memset(&params, 0, sizeof(params));

  params.is_bf16 = q.dtype() == torch::kBFloat16;

  // All stride are in elements, not bytes.
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.o_row_stride = q.stride(-3);
  params.o_head_stride = q.stride(-2);

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.d = d;
  params.d_rounded = d_rounded;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to
  // float to compare. [Minor] We want to round down since when we do the
  // comparison we use <= instead of <
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  TORCH_CHECK(p_dropout < 1.f);

  params.is_causal = is_causal;
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;
  params.is_seqlens_k_cumulative = true;

  // TODO(wenting.swt): support max_seqlen_q==1
  // if (seqlenq_ngroups_swapped) {}
  params.num_splits = 0;
  params.alibi_slopes_batch_stride = alibi_slopes_batch_stride;
  params.enable_alibi_slopes = enable_alibi_slopes;
}

void set_backward_params(
    FlashAttentionBackwardParams& params, const size_t b, const size_t seqlen_q,
    const size_t seqlen_k, const size_t h, const size_t h_k,
    const size_t d, const size_t d_rounded, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& dout,
    float p_dropout, float softmax_scale, bool is_causal, int window_size_left,
    int window_size_right, bool deterministic, int alibi_slopes_batch_stride,
    bool enable_alibi_slopes) {
  // Reset the parameters
  memset(&params, 0, sizeof(params));

  set_forward_params(params, b, seqlen_q, seqlen_k, h, h_k, d, d_rounded,
                     q, k, v, p_dropout, softmax_scale,
                     is_causal, window_size_left, window_size_right,
                     alibi_slopes_batch_stride, enable_alibi_slopes, false);
  // TODO(wenting.swt): check me, what if cu_seqlen_q is nullptr
  params.do_row_stride = dout.stride(-3);
  params.do_head_stride = dout.stride(-2);
  params.dq_row_stride = q.stride(-3);
  params.dk_row_stride = k.stride(-3);
  params.dv_row_stride = v.stride(-3);
  params.dq_head_stride = q.stride(-2);
  params.dk_head_stride = k.stride(-2);
  params.dv_head_stride = v.stride(-2);
  params.deterministic = deterministic;
}

FlashAttentionForwardParams get_flash_attention_forward_params(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    c10::optional<at::Tensor>& attention_mask, // (batch_size, seqlen)
    c10::optional<at::Tensor>& alibi_slopes_, const float p_dropout, const float softmax_scale,
    const bool zero_tensors, const bool is_causal, int window_size_left,
    int window_size_right, const bool return_softmax) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
  bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
  TORCH_CHECK(is_sm90 || is_sm8x,
              "FlashAttention only supports Ampere GPUs or newer.");

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 ||
              ((is_sm8x || is_sm90) && q_dtype == torch::kBFloat16));
  TORCH_CHECK(k.dtype() == q_dtype);
  TORCH_CHECK(v.dtype() == q_dtype);


  TORCH_CHECK(q.stride(-1) == 1);
  TORCH_CHECK(k.stride(-1) == 1);
  TORCH_CHECK(v.stride(-1) == 1);

  const auto sizes = q.sizes();
  const int batch_size = sizes[0];
  const int seqlen_q = sizes[1];
  const int num_heads = sizes[2];
  const int head_size_og = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);

  // TODO(wenting.swt): support max_seqlen_q==1
  // if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) {}

  TORCH_CHECK(batch_size > 0);
  TORCH_CHECK(head_size_og <= 256);
  TORCH_CHECK(num_heads % num_heads_k == 0);


  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);
  if (attention_mask.has_value()) {
    TORCH_CHECK(attention_mask.value().dtype() == torch::kInt32);
    CHECK_SHAPE(attention_mask.value(), batch_size, seqlen_k);
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);

  // User should pad in python
  TORCH_CHECK(head_size_og % 8 == 0);

  int alibi_slopes_batch_stride = 0;
  bool enable_alibi_slopes = false;
  if (alibi_slopes_.has_value()) {
    auto alibi_slopes = alibi_slopes_.value();
    TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32,
                "ALiBi slopes must have dtype fp32");
    TORCH_CHECK(alibi_slopes.stride(-1) == 1,
                "ALiBi slopes tensor must have contiguous last dimension");
    TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) ||
                alibi_slopes.sizes() ==
                    torch::IntArrayRef({batch_size, num_heads}));
    alibi_slopes_batch_stride =
        alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    enable_alibi_slopes = true;
  }

  FlashAttentionForwardParams params;
  set_forward_params(
      params, batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k, head_size, head_size_rounded, q, k, v, p_dropout,
      softmax_scale, is_causal, window_size_left, window_size_right,
      alibi_slopes_batch_stride, enable_alibi_slopes, /*seqlenq_ngroups_swapped*/false);

  return params;
}

FlashAttentionBackwardParams get_flash_attention_backward_params(
    const at::Tensor& dout, const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& out, const at::Tensor& softmax_lse,
    c10::optional<at::Tensor>& cu_seqlens_q, c10::optional<at::Tensor>& cu_seqlens_k,
    c10::optional<at::Tensor>& alibi_slopes_, const float p_dropout, const float softmax_scale,
    const bool zero_tensors, const bool is_causal, int window_size_left,
    int window_size_right, const bool deterministic) {
  if (is_causal) {
    window_size_right = 0;
  }
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
  bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
  bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
  TORCH_CHECK(is_sm90 || is_sm8x);

  bool is_dropout = p_dropout > 0.0;

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 ||
              ((is_sm8x || is_sm90) && q_dtype == torch::kBFloat16));
  TORCH_CHECK(k.dtype() == q_dtype);
  TORCH_CHECK(v.dtype() == q_dtype);
  TORCH_CHECK(out.dtype() == q_dtype);
  TORCH_CHECK(dout.dtype() == q_dtype);

  TORCH_CHECK(q.stride(-1) == 1);
  TORCH_CHECK(k.stride(-1) == 1);
  TORCH_CHECK(v.stride(-1) == 1);
  TORCH_CHECK(out.stride(-1) == 1);
  TORCH_CHECK(dout.stride(-1) == 1);

  const auto sizes = q.sizes();
  const int batch_size = sizes[0];
  const int seqlen_q = sizes[1];
  const int num_heads = sizes[2];
  const int head_size_og = dout.size(3);
  const int head_size = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);
  TORCH_CHECK(batch_size > 0);
  TORCH_CHECK((head_size % 8 == 0) && (head_size <= 256));
  if (head_size > 192) {
    TORCH_CHECK(is_sm80 || is_sm90,
                "FlashAttention backward for head dim > 192 requires A100/A800 "
                "or H100/H800");
  }

  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);

  TORCH_CHECK(head_size == round_multiple(head_size_og, 8),
              "head_size must be head_size_og rounded to a multiple of 8");

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size);

  if (cu_seqlens_q.has_value() && cu_seqlens_k.has_value()) {
    TORCH_CHECK(cu_seqlens_q.value().dtype() == torch::kInt32);
    TORCH_CHECK(cu_seqlens_k.value().dtype() == torch::kInt32);
    TORCH_CHECK(cu_seqlens_q.value().is_contiguous());
    TORCH_CHECK(cu_seqlens_k.value().is_contiguous());
    TORCH_CHECK(batch_size == cu_seqlens_q.value().numel() - 1);
    CHECK_SHAPE(cu_seqlens_q.value(), batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k.value(), batch_size + 1);
  }
  
  int alibi_slopes_batch_stride = 0;
  bool enable_alibi_slopes = false;
  if (alibi_slopes_.has_value()) {
    auto alibi_slopes = alibi_slopes_.value();
    TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32,
                "ALiBi slopes must have dtype fp32");
    TORCH_CHECK(alibi_slopes.stride(-1) == 1,
                "ALiBi slopes tensor must have contiguous last dimension");
    TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) ||
                alibi_slopes.sizes() ==
                    torch::IntArrayRef({batch_size, num_heads}));
    alibi_slopes_batch_stride =
        alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    enable_alibi_slopes = true;
  }

  FlashAttentionBackwardParams params;
  set_backward_params(
      params, batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k, head_size, head_size_rounded, q, k, v, dout, p_dropout,
      softmax_scale, is_causal, window_size_left, window_size_right,
      deterministic, alibi_slopes_batch_stride, enable_alibi_slopes);
  return params;
}

at::Tensor cu_seqlens_to_indices(const at::Tensor& cu_seqlens,
                                 int batch_size,
                                 int seqlen,
                                 torch::Dtype scalar_type,
                                 int& max_seqlen_in_batch,
                                 int& total) {
  total = cu_seqlens[-1].item<int>();
  torch::Tensor nonzero_counts = cu_seqlens.slice(
    0, 1, cu_seqlens.size(0)) - cu_seqlens.slice(
      0, 0, cu_seqlens.size(0) - 1);
  std::array<int64_t, 2> shape = {batch_size, seqlen};

  auto opts = torch::TensorOptions().dtype(scalar_type).device(torch::kCUDA);
  torch::Tensor rows = torch::arange(shape[0], opts.dtype(torch::kInt32)).unsqueeze(1);
  torch::Tensor cols = torch::arange(shape[1], opts.dtype(torch::kInt32)).unsqueeze(0);
  torch::Tensor mask = cols < nonzero_counts.unsqueeze(1);
  max_seqlen_in_batch = torch::sum(mask, {1}).max().item<int>();

  torch::Tensor matrix = torch::zeros(shape, opts.dtype(torch::kInt32));
  matrix.index_put_({mask}, torch::arange(
    1, mask.sum().item<int64_t>() + 1, opts.dtype(torch::kInt32)));
  torch::Tensor flattened_matrix = matrix.flatten();

  auto indices = torch::nonzero(flattened_matrix).squeeze();
  total = indices.size(0);
  return indices;
}

at::Tensor mask_to_indices(const at::Tensor& attention_mask,
                           int& max_seqlen_in_batch,
                           int& total,
                           at::Tensor& cu_seqlen) {
  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  at::Tensor seqlens_in_batch = torch::sum(attention_mask, {1});
  at::Tensor indices = torch::nonzero(attention_mask.flatten()).flatten();
  total = indices.size(0);
  max_seqlen_in_batch = seqlens_in_batch.max().item<int>();
  at::Tensor cumsum = torch::cumsum(seqlens_in_batch, 0);
  cu_seqlen.slice(0, 1, seqlens_in_batch.size(0) + 1).copy_(cumsum);
  return indices;
}

at::Tensor index_first_axis(const at::Tensor& input, const at::Tensor& indices) {
  torch::IntArrayRef sizes = input.sizes();
  int64_t first_axis_dim = sizes[0];
  auto other_shape = sizes.slice(1, sizes.size() - 1);

  int64_t second_dim = 1;
  for (auto dim: other_shape) {
    second_dim *= dim;
  }

  at::Tensor flat_input = torch::flatten(input, 1);
  torch::Tensor repeated_indices = indices.unsqueeze(1).expand({indices.size(0), second_dim});
  at::Tensor gather_input = torch::gather(flat_input, 0, repeated_indices);
  std::vector<int64_t> reshaped_size = {-1};
  reshaped_size.insert(reshaped_size.end(), other_shape.begin(), other_shape.end());
  return gather_input.reshape(reshaped_size);
}

}  // namespace torch_xla
