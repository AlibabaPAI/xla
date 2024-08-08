#include <c10/cuda/CUDAStream.h>
#include <ral/context/base/cuda/cuda_context_impl.h>
#include <ral/context/context_util.h>
#include <ral/device/gpu/gpu_driver.h>
#include <ral/ral_api.h>
#include <ral/ral_context.h>
#include <ral/ral_driver.h>
#include <ral/ral_helper.h>
#include <ral/ral_logging.h>
#include <torch/torch.h>

#include <Eigen/Core>
#include <algorithm>

#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "cutlass/numeric_types.h"
#include "flash.h"
#include "mlir/ral/context/pdll_util.h"
#include "mlir/ral/context/stream_executor_based_impl.h"
#include "static_switch.h"
#include "torch_xla/csrc/runtime/tf_logging.h"

namespace tao {
namespace ral {

DEFINE_TAO_TYPE_NAME_HELPER(Eigen::half, "f16");

struct FlashAttentionBackwardParams {
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

  // The stride between rows of O.
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  // The dimensions.
  int b, seqlen_q, seqlen_k, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded;

  int total_q;
  int total_k;

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

  // Backward specific params
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

  void FromString(const std::string& str) {
    std::vector<std::string> params_list = absl::StrSplit(str, "|");
    TORCH_CHECK(params_list.size() == 51);

    // Forward specific param
    absl::SimpleAtoi(params_list[0], &this->q_batch_stride);
    absl::SimpleAtoi(params_list[1], &this->k_batch_stride);
    absl::SimpleAtoi(params_list[2], &this->v_batch_stride);
    absl::SimpleAtoi(params_list[3], &this->q_row_stride);
    absl::SimpleAtoi(params_list[4], &this->k_row_stride);
    absl::SimpleAtoi(params_list[5], &this->v_row_stride);
    absl::SimpleAtoi(params_list[6], &this->q_head_stride);
    absl::SimpleAtoi(params_list[7], &this->k_head_stride);
    absl::SimpleAtoi(params_list[8], &this->v_head_stride);
    absl::SimpleAtoi(params_list[9], &this->total_q);
    absl::SimpleAtoi(params_list[10], &this->total_k);
    absl::SimpleAtoi(params_list[11], &this->h);
    absl::SimpleAtoi(params_list[12], &this->h_k);
    absl::SimpleAtoi(params_list[13], &this->h_h_k_ratio);
    absl::SimpleAtoi(params_list[14], &this->o_batch_stride);
    absl::SimpleAtoi(params_list[15], &this->o_row_stride);
    absl::SimpleAtoi(params_list[16], &this->o_head_stride);
    absl::SimpleAtoi(params_list[17], &this->b);
    absl::SimpleAtoi(params_list[18], &this->seqlen_q);
    absl::SimpleAtoi(params_list[19], &this->seqlen_k);
    absl::SimpleAtoi(params_list[20], &this->d);
    absl::SimpleAtoi(params_list[21], &this->seqlen_q_rounded);
    absl::SimpleAtoi(params_list[22], &this->seqlen_k_rounded);
    absl::SimpleAtoi(params_list[23], &this->d_rounded);
    absl::SimpleAtof(params_list[24], &this->scale_softmax);
    absl::SimpleAtof(params_list[25], &this->scale_softmax_log2);
    absl::SimpleAtof(params_list[26], &this->p_dropout);
    uint32_t tmp;
    absl::SimpleAtoi(params_list[27], &tmp);
    this->p_dropout_in_uint8_t = uint8_t(tmp);
    absl::SimpleAtof(params_list[28], &this->rp_dropout);
    absl::SimpleAtof(params_list[29], &this->scale_softmax_rp_dropout);
    absl::SimpleAtob(params_list[30], &this->is_bf16);
    absl::SimpleAtob(params_list[31], &this->is_causal);
    absl::SimpleAtoi(params_list[32], &this->window_size_left);
    absl::SimpleAtoi(params_list[33], &this->window_size_right);
    absl::SimpleAtoi(params_list[34], &this->alibi_slopes_batch_stride);
    absl::SimpleAtob(params_list[35], &this->is_seqlens_k_cumulative);
    absl::SimpleAtoi(params_list[36], &this->num_splits);
    absl::SimpleAtob(params_list[37], &this->enable_alibi_slopes);

    // backward specific params
    const int offset = 38;  // FlashAttentionForwardParams has 38 variables
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
};

void run_mha_bwd(Flash_bwd_params& params, cudaStream_t stream,
                 const bool configure) {
  FP16_SWITCH(!params.is_bf16, [&] {
    HEADDIM_SWITCH(params.d,
                   [&] { run_mha_bwd_<elem_type, kHeadDim>(params, stream); });
  });
}

// Layout of `buffers` listed above:
//  buffers[0] = dout
//  buffers[1] = q
//  buffers[2] = k
//  buffers[3] = v
//  buffers[4] = out
//  buffers[5] = softmax_lse
//  buffers[6] = cu_seqlens_q
//  buffers[7] = cu_seqlens_k
//  buffers[8] = rng_state
//  buffers[9] = alibi_slopes
//  buffers[10] = dq  // this is output
//  buffers[11] = dk  // this is output
//  buffers[12] = dv  // this is output
//  buffers[13] = softmax_d  // this is output
template <typename T_IN, typename SOFT_MAX_TYPE, int M>
std::tuple<MemRefType<T_IN, M>, MemRefType<T_IN, M>, MemRefType<T_IN, M>,
           MemRefType<SOFT_MAX_TYPE, M>>
custom_call_flash_attention_backward_impl(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> dout,
    MemRefType<T_IN, M> q, MemRefType<T_IN, M> k, MemRefType<T_IN, M> v,
    MemRefType<T_IN, M> out, MemRefType<SOFT_MAX_TYPE, M> softmax_lse,
    MemRefType<int32_t, 1> seqlens_q, MemRefType<int32_t, 1> seqlens_k,
    MemRefType<uint64_t, 1> rng_state, void* alibi_slopes_ptr,
    void* customAttrs) {
  auto attr = getOrParsePDLAttr(ctx, customAttrs,
                                "custom_call_flash_attention_backward");
  if (!attr) {
    ctx->signalError(Context::FAILURE, "fail to parse custom_attrs\n");
  }
  auto& dictAttr = attr->as<DictPDLAttr>();
  std::string backend_config =
      dictAttr.get("backend_config").template as<StrPDLAttr>().getValue();

  auto gpu_driver = ctx->getDriver<tao::ral::gpu::GPUDriver>(
      tao::ral::gpu::GPUDriver::name());
  auto gpu_stream =
      static_cast<cudaStream_t>(gpu_driver->asCUStream(ctx, stream_handle));

  int softmax_element_count = 1, q_element_count = 1, k_element_count = 1,
      v_element_count = 1;
  for (int i = 0; i < M; i++) {
    q_element_count *= q.sizes[i];
    k_element_count *= k.sizes[i];
    v_element_count *= v.sizes[i];
    softmax_element_count *= softmax_lse.sizes[i];
  }

  auto dq_ptr = static_cast<T_IN*>(
      gpu_driver->alloc(ctx, q_element_count * sizeof(T_IN)));
  auto dq_res = assignMemRef<T_IN, M>(dq_ptr, q.sizes);

  auto dk_ptr = static_cast<T_IN*>(
      gpu_driver->alloc(ctx, k_element_count * sizeof(T_IN)));
  auto dk_res = assignMemRef<T_IN, M>(dk_ptr, k.sizes);

  auto dv_ptr = static_cast<T_IN*>(
      gpu_driver->alloc(ctx, v_element_count * sizeof(T_IN)));
  auto dv_res = assignMemRef<T_IN, M>(dv_ptr, v.sizes);

  auto dsoftmax_ptr = static_cast<SOFT_MAX_TYPE*>(
      gpu_driver->alloc(ctx, softmax_element_count * sizeof(SOFT_MAX_TYPE)));
  auto dsoftmax =
      assignMemRef<SOFT_MAX_TYPE, M>(dsoftmax_ptr, softmax_lse.sizes);

  FlashAttentionBackwardParams params;
  params.FromString(std::move(backend_config));
  Flash_bwd_params launch_params;

  // Reset the parameters
  memset(&launch_params, 0, sizeof(launch_params));

  launch_params.is_bf16 = params.is_bf16;

  // Set the pointers and strides.
  launch_params.q_ptr = q.data;
  launch_params.k_ptr = k.data;
  launch_params.v_ptr = v.data;
  // All stride are in elements, not bytes.
  launch_params.q_row_stride = params.q_row_stride;
  launch_params.k_row_stride = params.k_row_stride;
  launch_params.v_row_stride = params.v_row_stride;
  launch_params.q_head_stride = params.q_head_stride;
  launch_params.k_head_stride = params.k_head_stride;
  launch_params.v_head_stride = params.v_head_stride;
  launch_params.o_ptr = out.data;
  launch_params.o_row_stride = params.o_row_stride;
  launch_params.o_head_stride = params.o_head_stride;

  launch_params.cu_seqlens_q = static_cast<int*>(seqlens_q.data);
  launch_params.cu_seqlens_k = static_cast<int*>(seqlens_k.data);

  launch_params.alibi_slopes_ptr = alibi_slopes_ptr;
  launch_params.alibi_slopes_batch_stride = params.alibi_slopes_batch_stride;

  // P = softmax(QK^T)
  launch_params.p_ptr = nullptr;  // no softmax returned always

  // Softmax sum
  launch_params.softmax_lse_ptr = softmax_lse.data;

  // Set the dimensions.
  launch_params.b = params.b;
  launch_params.h = params.h;
  launch_params.h_k = params.h_k;
  launch_params.h_h_k_ratio = params.h_h_k_ratio;
  launch_params.seqlen_q = params.seqlen_q;
  launch_params.seqlen_k = params.seqlen_k;
  launch_params.seqlen_q_rounded = params.seqlen_q_rounded;
  launch_params.seqlen_k_rounded = params.seqlen_k_rounded;
  launch_params.d = params.d;
  launch_params.d_rounded = params.d_rounded;

  // Set the different scale values.
  launch_params.scale_softmax = params.scale_softmax;
  launch_params.scale_softmax_log2 = params.scale_softmax_log2;

  launch_params.p_dropout = params.p_dropout;
  launch_params.p_dropout_in_uint8_t = params.p_dropout_in_uint8_t;
  launch_params.rp_dropout = params.rp_dropout;
  launch_params.scale_softmax_rp_dropout = params.scale_softmax_rp_dropout;

  launch_params.is_causal = params.is_causal;
  launch_params.window_size_left = params.window_size_left;
  launch_params.window_size_right = params.window_size_right;

  launch_params.is_seqlens_k_cumulative = true;

  launch_params.do_ptr = dout.data;
  launch_params.do_row_stride = params.do_row_stride;
  launch_params.do_head_stride = params.do_head_stride;
  launch_params.dq_ptr = dq_res.data;
  launch_params.dk_ptr = dk_res.data;
  launch_params.dv_ptr = dv_res.data;
  launch_params.dq_row_stride = params.dq_row_stride;
  launch_params.dk_row_stride = params.dk_row_stride;
  launch_params.dv_row_stride = params.dv_row_stride;
  launch_params.dq_head_stride = params.dq_head_stride;
  launch_params.dk_head_stride = params.dk_head_stride;
  launch_params.dv_head_stride = params.dv_head_stride;

  // bool loop = max_seqlen_k > blocksize_c;
  // TODO: change later, for now set to true for simplicity
  bool loop = true;
  auto scalar_type = params.is_bf16 ? torch::kBFloat16 : torch::kFloat16;
  auto opts = torch::TensorOptions().dtype(scalar_type).device(torch::kCUDA);
  at::Tensor dq_accum;
  if (loop) {
    if (!params.deterministic) {
      dq_accum = torch::empty({params.total_q + 128 * launch_params.b,
                               launch_params.h, launch_params.d_rounded},
                              opts.dtype(at::kFloat));
    } else {
      auto dprops = at::cuda::getCurrentDeviceProperties();
      const int nsplits = (dprops->multiProcessorCount +
                           launch_params.b * launch_params.h - 1) /
                          (launch_params.b * launch_params.h);
      dq_accum = torch::zeros({nsplits, params.total_q + 128 * launch_params.b,
                               launch_params.h, launch_params.d_rounded},
                              opts.dtype(at::kFloat));
    }
  }

  at::Tensor dk = torch::from_blob(
      dk_res.data, {params.total_k, launch_params.h_k, launch_params.d}, opts);
  at::Tensor dv = torch::from_blob(
      dv_res.data, {params.total_k, launch_params.h_k, launch_params.d}, opts);

  at::Tensor dk_expanded, dv_expanded;

  if (launch_params.h_k != launch_params.h) {  // MQA / GQA
    TF_VLOG(2) << "Running FlashAttention Backward as MQA/GQA";
    dk_expanded =
        torch::empty({params.total_k, launch_params.h, launch_params.d}, opts);
    dv_expanded =
        torch::empty({params.total_k, launch_params.h, launch_params.d}, opts);

    launch_params.dk_ptr = dk_expanded.data_ptr();
    launch_params.dv_ptr = dv_expanded.data_ptr();
    launch_params.dk_row_stride = dk_expanded.stride(-3);
    launch_params.dv_row_stride = dv_expanded.stride(-3);
    launch_params.dk_head_stride = dk_expanded.stride(-2);
    launch_params.dv_head_stride = dv_expanded.stride(-2);
  } else {
    TF_VLOG(2) << "Running FlashAttention Backward";
    dk_expanded = dk;
    dv_expanded = dv;
  }

  launch_params.dq_accum_ptr = loop ? dq_accum.data_ptr() : nullptr;
  launch_params.dk_accum_ptr = nullptr;
  launch_params.dv_accum_ptr = nullptr;

  // Softmax sum
  launch_params.dsoftmax_sum = dsoftmax.data;

  launch_params.deterministic = params.deterministic;
  launch_params.dq_accum_split_stride =
      !launch_params.deterministic ? 0 : dq_accum.stride(0);

  auto launch = &run_mha_bwd;

  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

  // We use a custom RNG that increases the offset by batch_size * nheads * 32.
  int64_t counter_offset = launch_params.b * launch_params.h * 32;

  bool is_dropout = (1.f - launch_params.p_dropout) > 0.0;
  // TODO(wenting.swt): According to the implementation in
  // `flash_attn_varlen_func` of flash-attn v2.5.6, the forward generates
  // `rng_state` which is passed as ctx to the backward. Hence, for simplifying
  // the logic, the redundant branch where `rng_state` is None has been omitted.
  launch_params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data);

  launch(launch_params, gpu_stream, /*configure=*/false);

  // For MQA/GQA we need to sum dK and dV across the groups
  if (launch_params.h_k != launch_params.h) {
    at::sum_out(dk,
                at::reshape(dk_expanded, {params.total_k, launch_params.h_k,
                                          launch_params.h / launch_params.h_k,
                                          launch_params.d}),
                {2});
    at::sum_out(dv,
                at::reshape(dv_expanded, {params.total_k, launch_params.h_k,
                                          launch_params.h / launch_params.h_k,
                                          launch_params.d}),
                {2});
  }

  return std::make_tuple(dq_res, dk_res, dv_res, dsoftmax);
}

template <typename T_IN, typename SOFT_MAX_TYPE, int M>
std::tuple<MemRefType<T_IN, M>, MemRefType<T_IN, M>, MemRefType<T_IN, M>,
           MemRefType<SOFT_MAX_TYPE, M>>
custom_call_flash_attention_backward_noalibi(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> dout,
    MemRefType<T_IN, M> q, MemRefType<T_IN, M> k, MemRefType<T_IN, M> v,
    MemRefType<T_IN, M> out, MemRefType<SOFT_MAX_TYPE, M> softmax_lse,
    MemRefType<int32_t, 1> seqlens_q, MemRefType<int32_t, 1> seqlens_k,
    MemRefType<uint64_t, 1> rng_state, void* customAttrs) {
  return custom_call_flash_attention_backward_impl<T_IN, SOFT_MAX_TYPE, M>(
      ctx, stream_handle, dout, q, k, v, out, softmax_lse, seqlens_q, seqlens_k,
      rng_state, nullptr, customAttrs);
}

template <typename T_IN, typename SOFT_MAX_TYPE, int M>
std::tuple<MemRefType<T_IN, M>, MemRefType<T_IN, M>, MemRefType<T_IN, M>,
           MemRefType<SOFT_MAX_TYPE, M>>
custom_call_flash_attention_backward_alibi_v1(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> dout,
    MemRefType<T_IN, M> q, MemRefType<T_IN, M> k, MemRefType<T_IN, M> v,
    MemRefType<T_IN, M> out, MemRefType<SOFT_MAX_TYPE, M> softmax_lse,
    MemRefType<int32_t, 1> seqlens_q, MemRefType<int32_t, 1> seqlens_k,
    MemRefType<uint64_t, 1> rng_state, MemRefType<float, 1> alibi_slopes,
    void* customAttrs) {
  return custom_call_flash_attention_backward_impl<T_IN, SOFT_MAX_TYPE, M>(
      ctx, stream_handle, dout, q, k, v, out, softmax_lse, seqlens_q, seqlens_k,
      rng_state, alibi_slopes.data, customAttrs);
}

template <typename T_IN, typename SOFT_MAX_TYPE, int M>
std::tuple<MemRefType<T_IN, M>, MemRefType<T_IN, M>, MemRefType<T_IN, M>,
           MemRefType<SOFT_MAX_TYPE, M>>
custom_call_flash_attention_backward_alibi_v2(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> dout,
    MemRefType<T_IN, M> q, MemRefType<T_IN, M> k, MemRefType<T_IN, M> v,
    MemRefType<T_IN, M> out, MemRefType<SOFT_MAX_TYPE, M> softmax_lse,
    MemRefType<int32_t, 1> seqlens_q, MemRefType<int32_t, 1> seqlens_k,
    MemRefType<uint64_t, 1> rng_state, MemRefType<float, 2> alibi_slopes,
    void* customAttrs) {
  return custom_call_flash_attention_backward_impl<T_IN, SOFT_MAX_TYPE, M>(
      ctx, stream_handle, dout, q, k, v, out, softmax_lse, seqlens_q, seqlens_k,
      rng_state, alibi_slopes.data, customAttrs);
}

TAO_RAL_API(
    "custom_call_flash_attention_backward", "gpu",
    custom_call_flash_attention_backward_noalibi<Eigen::half, float, 3>);
TAO_RAL_API(
    "custom_call_flash_attention_backward", "gpu",
    custom_call_flash_attention_backward_alibi_v1<Eigen::half, float, 3>);
TAO_RAL_API(
    "custom_call_flash_attention_backward", "gpu",
    custom_call_flash_attention_backward_alibi_v2<Eigen::half, float, 3>);
TAO_RAL_API("custom_call_flash_attention_backward", "gpu",
            custom_call_flash_attention_backward_noalibi<bfloat16, float, 3>);
TAO_RAL_API("custom_call_flash_attention_backward", "gpu",
            custom_call_flash_attention_backward_alibi_v1<bfloat16, float, 3>);
TAO_RAL_API("custom_call_flash_attention_backward", "gpu",
            custom_call_flash_attention_backward_alibi_v2<bfloat16, float, 3>);

}  // namespace ral
}  // namespace tao