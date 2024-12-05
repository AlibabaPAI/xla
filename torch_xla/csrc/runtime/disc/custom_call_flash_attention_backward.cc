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
#include "torch_xla/csrc/flash_attention_utils.h"
#include "torch_xla/csrc/runtime/tf_logging.h"

namespace tao {
namespace ral {

DEFINE_TAO_TYPE_NAME_HELPER(Eigen::half, "f16");

void run_mha_bwd(Flash_bwd_params& params, cudaStream_t stream,
                 const bool configure) {
  FP16_SWITCH(!params.is_bf16, [&] {
    HEADDIM_SWITCH(params.d,
                   [&] { run_mha_bwd_<elem_type, kHeadDim>(params, stream); });
  });
}

template <typename T_IN, int M>
std::tuple<MemRefType<T_IN, M>, MemRefType<T_IN, M>, MemRefType<T_IN, M>,
           MemRefType<float, 3>>
custom_call_flash_attention_varlen_backward_impl(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> dout_memref,
    MemRefType<T_IN, M> q_memref, MemRefType<T_IN, M> k_memref,
    MemRefType<T_IN, M> v_memref, MemRefType<T_IN, M> out_memref,
    MemRefType<float, 3> softmax_lse_memref,
    MemRefType<int32_t, 1> seqlens_q_memref,
    MemRefType<int32_t, 1> seqlens_k_memref, MemRefType<int64_t, 1> rng_state,
    void* alibi_slopes_ptr, void* customAttrs) {
  auto attr = getOrParsePDLAttr(ctx, customAttrs,
                                "custom_call_flash_attention_varlen_backward");
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
    q_element_count *= q_memref.sizes[i];
    k_element_count *= k_memref.sizes[i];
    v_element_count *= v_memref.sizes[i];
  }

  for (int i = 0; i < 3; i++) {
    softmax_element_count *= softmax_lse_memref.sizes[i];
  }

  auto dq_ptr = static_cast<T_IN*>(
      gpu_driver->alloc(ctx, q_element_count * sizeof(T_IN)));

  auto dk_ptr = static_cast<T_IN*>(
      gpu_driver->alloc(ctx, k_element_count * sizeof(T_IN)));

  auto dv_ptr = static_cast<T_IN*>(
      gpu_driver->alloc(ctx, v_element_count * sizeof(T_IN)));

  auto dsoftmax_ptr = static_cast<float*>(
      gpu_driver->alloc(ctx, softmax_element_count * sizeof(float)));

  torch_xla::FlashAttentionBackwardParams params;
  params.FromString(std::move(backend_config));

  int bs = q_memref.sizes[0];
  int seqlen_q = q_memref.sizes[1];
  int seqlen_k = k_memref.sizes[1];

  // For simplification, we do not currently support dynamic dimensions for head
  // and headdim here.
  if (params.h != q_memref.sizes[2]) {
    ctx->signalError(Context::FAILURE,
                     "Currently, it is not supported for the head dimension of "
                     "q to be dynamic.\n");
  }
  if (params.h_k != k_memref.sizes[2]) {
    ctx->signalError(Context::FAILURE,
                     "Currently, it is not supported for the head dimension of "
                     "k to be dynamic.\n");
  }
  if (params.d != q_memref.sizes[3]) {
    ctx->signalError(Context::FAILURE,
                     "Currently, it is not supported for the headdim dimension "
                     "of q to be dynamic.\n");
  }

  auto scalar_type = params.is_bf16 ? torch::kBFloat16 : torch::kFloat16;
  auto opts = torch::TensorOptions().dtype(scalar_type).device(torch::kCUDA);

  // Inputs
  at::Tensor do_ = torch::from_blob(dout_memref.data,
                                    {bs * seqlen_q, params.h, params.d}, opts);
  at::Tensor q = torch::from_blob(q_memref.data,
                                  {bs * seqlen_q, params.h, params.d}, opts);
  at::Tensor k = torch::from_blob(k_memref.data,
                                  {bs * seqlen_k, params.h_k, params.d}, opts);
  at::Tensor v = torch::from_blob(v_memref.data,
                                  {bs * seqlen_k, params.h_k, params.d}, opts);
  at::Tensor o = torch::from_blob(out_memref.data,
                                  {bs * seqlen_q, params.h, params.d}, opts);
  at::Tensor softmax_lse =
      torch::from_blob(softmax_lse_memref.data, {bs, params.h, seqlen_q},
                       opts.dtype(torch::kFloat));
  at::Tensor cu_seqlens_q = torch::from_blob(seqlens_q_memref.data, {bs + 1},
                                             opts.dtype(torch::kInt32));
  at::Tensor cu_seqlens_k = torch::from_blob(seqlens_k_memref.data, {bs + 1},
                                             opts.dtype(torch::kInt32));

  // Outputs
  at::Tensor dq =
      torch::from_blob(dq_ptr, {bs * seqlen_q, params.h, params.d}, opts);

  at::Tensor dk =
      torch::from_blob(dk_ptr, {bs * seqlen_k, params.h_k, params.d}, opts);

  at::Tensor dv =
      torch::from_blob(dv_ptr, {bs * seqlen_k, params.h_k, params.d}, opts);

  at::Tensor dsoftmax_sum = torch::from_blob(
      dsoftmax_ptr, {bs, params.h, seqlen_q}, opts.dtype(torch::kFloat));

  // Fill zeros for outputs.
  dq.fill_(0);
  dk.fill_(0);
  dv.fill_(0);
  dsoftmax_sum.fill_(0);

  int max_seqlen_in_batch_q = seqlen_q;
  int max_seqlen_in_batch_k = seqlen_k;
  int total_q = bs * seqlen_q;
  int total_k = bs * seqlen_k;
  at::Tensor indices_q = torch_xla::cu_seqlens_to_indices(
      cu_seqlens_q, bs, seqlen_q, scalar_type, max_seqlen_in_batch_q, total_q);
  at::Tensor indices_k;
  if (seqlen_q == seqlen_k) {
    indices_k = indices_q;
  } else {
    indices_k = torch_xla::cu_seqlens_to_indices(
        cu_seqlens_k, bs, seqlen_k, scalar_type, max_seqlen_in_batch_k,
        total_k);
  }

  // The unpaded inputs
  auto unpad_do = torch_xla::index_first_axis(do_, indices_q);
  auto unpad_q = torch_xla::index_first_axis(q, indices_q);
  auto unpad_k = torch_xla::index_first_axis(k, indices_k);
  auto unpad_v = torch_xla::index_first_axis(v, indices_k);
  auto unpad_o = torch_xla::index_first_axis(o, indices_q);
  auto unpad_softmax_lse =
      softmax_lse
          .index({torch::indexing::Slice(), torch::indexing::Slice(),
                  torch::indexing::Slice(0, max_seqlen_in_batch_q)})
          .contiguous();

  // The upaded outputs
  at::Tensor unpad_dq = at::zeros({total_q, params.h, params.d}, opts);
  at::Tensor unpad_dk = at::zeros({total_k, params.h_k, params.d}, opts);
  at::Tensor unpad_dv = at::zeros({total_k, params.h_k, params.d}, opts);
  // the upaded dsoftmax_lse will be inited later

  Flash_bwd_params launch_params;

  // Reset the parameters
  memset(&launch_params, 0, sizeof(launch_params));

  launch_params.is_bf16 = params.is_bf16;

  // Set the pointers and strides.
  launch_params.q_ptr = unpad_q.data_ptr();
  launch_params.k_ptr = unpad_k.data_ptr();
  launch_params.v_ptr = unpad_v.data_ptr();
  launch_params.o_ptr = unpad_o.data_ptr();

  // All stride are in elements, not bytes.
  launch_params.q_row_stride = params.q_row_stride;
  launch_params.k_row_stride = params.k_row_stride;
  launch_params.v_row_stride = params.v_row_stride;
  launch_params.q_head_stride = params.q_head_stride;
  launch_params.k_head_stride = params.k_head_stride;
  launch_params.v_head_stride = params.v_head_stride;
  launch_params.o_row_stride = params.o_row_stride;
  launch_params.o_head_stride = params.o_head_stride;

  launch_params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q.data_ptr());
  launch_params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k.data_ptr());
  launch_params.softmax_lse_ptr = unpad_softmax_lse.data_ptr();

  launch_params.alibi_slopes_ptr = alibi_slopes_ptr;

  launch_params.alibi_slopes_batch_stride = params.alibi_slopes_batch_stride;

  // P = softmax(QK^T)
  launch_params.p_ptr = nullptr;  // no softmax returned always

  // Set the dimensions.
  launch_params.b = bs;
  launch_params.h = params.h;
  launch_params.h_k = params.h_k;
  launch_params.h_h_k_ratio = params.h_h_k_ratio;
  launch_params.seqlen_q = max_seqlen_in_batch_q;
  launch_params.seqlen_k = max_seqlen_in_batch_k;
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  launch_params.seqlen_q_rounded = round_multiple(max_seqlen_in_batch_q, 128);
  launch_params.seqlen_k_rounded = round_multiple(max_seqlen_in_batch_k, 128);
  launch_params.d = params.d;
  launch_params.d_rounded = params.d_rounded;

  // Set the different scale values.
  launch_params.scale_softmax = params.scale_softmax;
  launch_params.scale_softmax_log2 = params.scale_softmax_log2;

  launch_params.p_dropout = params.p_dropout;
  launch_params.p_dropout_in_uint8_t = params.p_dropout_in_uint8_t;
  launch_params.rp_dropout = params.rp_dropout;
  launch_params.scale_softmax_rp_dropout = params.scale_softmax_rp_dropout;

  if (max_seqlen_in_batch_q == 1) {
    params.is_causal = false;
  }
  if (params.is_causal) {
    params.window_size_right = 0;
  }

  if (params.window_size_left >= max_seqlen_in_batch_k) {
    params.window_size_left = -1;
  }
  if (params.window_size_right >= max_seqlen_in_batch_k) {
    params.window_size_right = -1;
  }

  launch_params.is_causal =
      params.window_size_left < 0 && params.window_size_right == 0;

  if (params.window_size_left < 0 && params.window_size_right >= 0) {
    params.window_size_left = max_seqlen_in_batch_k;
  }
  if (params.window_size_left >= 0 && params.window_size_right < 0) {
    params.window_size_right = max_seqlen_in_batch_k;
  }

  launch_params.window_size_left = params.window_size_left;
  launch_params.window_size_right = params.window_size_right;

  launch_params.is_seqlens_k_cumulative = true;

  launch_params.do_row_stride = params.do_row_stride;
  launch_params.do_head_stride = params.do_head_stride;

  launch_params.dq_row_stride = params.dq_row_stride;
  launch_params.dk_row_stride = params.dk_row_stride;
  launch_params.dv_row_stride = params.dv_row_stride;
  launch_params.dq_head_stride = params.dq_head_stride;
  launch_params.dk_head_stride = params.dk_head_stride;
  launch_params.dv_head_stride = params.dv_head_stride;

  at::Tensor rounded_dsoftmax_sum =
      at::zeros({bs, params.h, launch_params.seqlen_q_rounded},
                opts.dtype(torch::kFloat));

  launch_params.do_ptr = unpad_do.data_ptr();
  launch_params.dq_ptr = unpad_dq.data_ptr();
  launch_params.dk_ptr = unpad_dk.data_ptr();
  launch_params.dv_ptr = unpad_dv.data_ptr();
  launch_params.dsoftmax_sum = rounded_dsoftmax_sum.data_ptr();

  // bool loop = max_seqlen_k > blocksize_c;
  // TODO: change later, for now set to true for simplicity
  bool loop = true;

  at::Tensor dq_accum;
  if (loop) {
    if (!params.deterministic) {
      dq_accum = torch::empty({total_q + 128 * launch_params.b, launch_params.h,
                               launch_params.d_rounded},
                              opts.dtype(at::kFloat));
    } else {
      auto dprops = at::cuda::getCurrentDeviceProperties();
      const int nsplits = (dprops->multiProcessorCount +
                           launch_params.b * launch_params.h - 1) /
                          (launch_params.b * launch_params.h);
      dq_accum = torch::zeros({nsplits, total_q + 128 * launch_params.b,
                               launch_params.h, launch_params.d_rounded},
                              opts.dtype(at::kFloat));
    }
  }

  at::Tensor dk_expanded, dv_expanded;

  if (launch_params.h_k != launch_params.h) {  // MQA / GQA
    TF_VLOG(2) << "Running FlashAttention Backward as MQA/GQA";
    dk_expanded =
        torch::empty({total_k, launch_params.h, launch_params.d}, opts);
    dv_expanded =
        torch::empty({total_k, launch_params.h, launch_params.d}, opts);

    launch_params.dk_ptr = dk_expanded.data_ptr();
    launch_params.dv_ptr = dv_expanded.data_ptr();
    launch_params.dk_row_stride = dk_expanded.stride(-3);
    launch_params.dv_row_stride = dv_expanded.stride(-3);
    launch_params.dk_head_stride = dk_expanded.stride(-2);
    launch_params.dv_head_stride = dv_expanded.stride(-2);
  } else {
    TF_VLOG(2) << "Running FlashAttention Backward";
    dk_expanded = unpad_dk;
    dv_expanded = unpad_dv;
  }

  launch_params.dq_accum_ptr = loop ? dq_accum.data_ptr() : nullptr;
  launch_params.dk_accum_ptr = nullptr;
  launch_params.dv_accum_ptr = nullptr;

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
    at::sum_out(unpad_dk,
                at::reshape(dk_expanded, {total_k, launch_params.h_k,
                                          launch_params.h / launch_params.h_k,
                                          launch_params.d}),
                {2});
    at::sum_out(unpad_dv,
                at::reshape(dv_expanded, {total_k, launch_params.h_k,
                                          launch_params.h / launch_params.h_k,
                                          launch_params.d}),
                {2});
  }

  torch::Tensor repeated_indices_q = indices_q.unsqueeze(1).unsqueeze(1).expand(
      {indices_q.size(0), params.h, params.d});
  torch::Tensor repeated_indices_k = indices_k.unsqueeze(1).unsqueeze(1).expand(
      {indices_k.size(0), params.h_k, params.d});

  dq.scatter_(0, repeated_indices_q, unpad_dq);
  dk.scatter_(0, repeated_indices_k, unpad_dk);
  dv.scatter_(0, repeated_indices_k, unpad_dv);
  dsoftmax_sum.slice(2, 0, max_seqlen_in_batch_q)
      .copy_(rounded_dsoftmax_sum.slice(2, 0, max_seqlen_in_batch_q));

  auto dq_memref = assignMemRef<T_IN, M>(dq_ptr, q_memref.sizes);
  auto dk_memref = assignMemRef<T_IN, M>(dk_ptr, k_memref.sizes);
  auto dv_memref = assignMemRef<T_IN, M>(dv_ptr, v_memref.sizes);
  auto dsoftmax_memref =
      assignMemRef<float, 3>(dsoftmax_ptr, softmax_lse_memref.sizes);

  return std::make_tuple(dq_memref, dk_memref, dv_memref, dsoftmax_memref);
}

template <typename T_IN, int M>
std::tuple<MemRefType<T_IN, M>, MemRefType<T_IN, M>, MemRefType<T_IN, M>,
           MemRefType<float, 3>>
custom_call_flash_attention_varlen_backward_noalibi(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> dout,
    MemRefType<T_IN, M> q, MemRefType<T_IN, M> k, MemRefType<T_IN, M> v,
    MemRefType<T_IN, M> out, MemRefType<float, 3> softmax_lse,
    MemRefType<int32_t, 1> seqlens_q, MemRefType<int32_t, 1> seqlens_k,
    MemRefType<int64_t, 1> rng_state, void* customAttrs) {
  return custom_call_flash_attention_varlen_backward_impl<T_IN, M>(
      ctx, stream_handle, dout, q, k, v, out, softmax_lse, seqlens_q, seqlens_k,
      rng_state, nullptr, customAttrs);
}

template <typename T_IN, int M>
std::tuple<MemRefType<T_IN, M>, MemRefType<T_IN, M>, MemRefType<T_IN, M>,
           MemRefType<float, 3>>
custom_call_flash_attention_varlen_backward_alibi_v1(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> dout,
    MemRefType<T_IN, M> q, MemRefType<T_IN, M> k, MemRefType<T_IN, M> v,
    MemRefType<T_IN, M> out, MemRefType<float, 3> softmax_lse,
    MemRefType<int32_t, 1> seqlens_q, MemRefType<int32_t, 1> seqlens_k,
    MemRefType<int64_t, 1> rng_state, MemRefType<float, 1> alibi_slopes,
    void* customAttrs) {
  return custom_call_flash_attention_varlen_backward_impl<T_IN, M>(
      ctx, stream_handle, dout, q, k, v, out, softmax_lse, seqlens_q, seqlens_k,
      rng_state, alibi_slopes.data, customAttrs);
}

template <typename T_IN, int M>
std::tuple<MemRefType<T_IN, M>, MemRefType<T_IN, M>, MemRefType<T_IN, M>,
           MemRefType<float, 3>>
custom_call_flash_attention_varlen_backward_alibi_v2(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> dout,
    MemRefType<T_IN, M> q, MemRefType<T_IN, M> k, MemRefType<T_IN, M> v,
    MemRefType<T_IN, M> out, MemRefType<float, 3> softmax_lse,
    MemRefType<int32_t, 1> seqlens_q, MemRefType<int32_t, 1> seqlens_k,
    MemRefType<int64_t, 1> rng_state, MemRefType<float, 2> alibi_slopes,
    void* customAttrs) {
  return custom_call_flash_attention_varlen_backward_impl<T_IN, M>(
      ctx, stream_handle, dout, q, k, v, out, softmax_lse, seqlens_q, seqlens_k,
      rng_state, alibi_slopes.data, customAttrs);
}

TAO_RAL_API(
    "custom_call_flash_attention_varlen_backward", "gpu",
    custom_call_flash_attention_varlen_backward_noalibi<Eigen::half, 4>);
TAO_RAL_API(
    "custom_call_flash_attention_varlen_backward", "gpu",
    custom_call_flash_attention_varlen_backward_alibi_v1<Eigen::half, 4>);
TAO_RAL_API(
    "custom_call_flash_attention_varlen_backward", "gpu",
    custom_call_flash_attention_varlen_backward_alibi_v2<Eigen::half, 4>);
TAO_RAL_API("custom_call_flash_attention_varlen_backward", "gpu",
            custom_call_flash_attention_varlen_backward_noalibi<bfloat16, 4>);
TAO_RAL_API("custom_call_flash_attention_varlen_backward", "gpu",
            custom_call_flash_attention_varlen_backward_alibi_v1<bfloat16, 4>);
TAO_RAL_API("custom_call_flash_attention_varlen_backward", "gpu",
            custom_call_flash_attention_varlen_backward_alibi_v2<bfloat16, 4>);

}  // namespace ral
}  // namespace tao