#include "torch_xla/csrc/ops/flash_attention_varlen_backward.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "cutlass/numeric_types.h"
#include "flash.h"
#include "static_switch.h"
#include "torch_xla/csrc/flash_attention_utils.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/xla_lower_util.h"
#include "xla/service/custom_call_target_registry.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& q,
                           const torch::lazy::Value& k,
                           const torch::lazy::Value& v,
                           const torch::lazy::Value& softmax_lse) {
  return xla::ShapeUtil::MakeTupleShape({GetXlaShape(q), GetXlaShape(k),
                                         GetXlaShape(v),
                                         GetXlaShape(softmax_lse)});
}

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
void custom_call_flash_attention_varlen_backward(cudaStream_t stream,
                                                 void** buffers,
                                                 const char* opaque,
                                                 size_t opaque_len) {
  std::string opaque_str(opaque, opaque_len);
  TF_VLOG(3) << "custom_call_flash_attention_varlen_backward opaque str: "
             << opaque_str;
  FlashAttentionBackwardParams params;
  params.FromString(std::move(opaque_str));
  int buf_offset = params.enable_alibi_slopes;
  auto scalar_type = params.is_bf16 ? torch::kBFloat16 : torch::kFloat16;

  cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream().stream();
  cudaEvent_t torch_wait_xla_event;
  cudaEventCreateWithFlags(&torch_wait_xla_event, cudaEventDisableTiming);
  cudaEvent_t xla_wait_torch_event;
  cudaEventCreateWithFlags(&xla_wait_torch_event, cudaEventDisableTiming);
  cudaEventRecord(torch_wait_xla_event, stream);
  cudaStreamWaitEvent(torch_stream, torch_wait_xla_event);

  auto cuda_stream = at::cuda::getDefaultCUDAStream();
  at::cuda::CUDAStreamGuard guard(cuda_stream);

  auto opts = torch::TensorOptions().dtype(scalar_type).device(torch::kCUDA);

  // Inputs
  at::Tensor do_ = torch::from_blob(
      buffers[0], {params.b * params.seqlen_q, params.h, params.d}, opts);
  at::Tensor q = torch::from_blob(
      buffers[1], {params.b * params.seqlen_q, params.h, params.d}, opts);
  at::Tensor k = torch::from_blob(
      buffers[2], {params.b * params.seqlen_k, params.h_k, params.d}, opts);
  at::Tensor v = torch::from_blob(
      buffers[3], {params.b * params.seqlen_k, params.h_k, params.d}, opts);
  at::Tensor o = torch::from_blob(
      buffers[4], {params.b * params.seqlen_q, params.h, params.d}, opts);
  at::Tensor softmax_lse =
      torch::from_blob(buffers[5], {params.b, params.h, params.seqlen_q},
                       opts.dtype(torch::kFloat));
  at::Tensor cu_seqlens_q =
      torch::from_blob(buffers[6], {params.b + 1}, opts.dtype(torch::kInt32));
  at::Tensor cu_seqlens_k =
      torch::from_blob(buffers[7], {params.b + 1}, opts.dtype(torch::kInt32));

  // Outputs
  at::Tensor dq =
      torch::from_blob(buffers[9 + buf_offset],
                       {params.b * params.seqlen_q, params.h, params.d}, opts);

  at::Tensor dk = torch::from_blob(
      buffers[10 + buf_offset],
      {params.b * params.seqlen_k, params.h_k, params.d}, opts);

  at::Tensor dv = torch::from_blob(
      buffers[11 + buf_offset],
      {params.b * params.seqlen_k, params.h_k, params.d}, opts);

  at::Tensor dsoftmax_sum = torch::from_blob(
      buffers[12 + buf_offset], {params.b, params.h, params.seqlen_q},
      opts.dtype(torch::kFloat));

  // Fill zeros for outputs.
  // cudaMemsetAsync(buffers[9 + buf_offset], 0,params.b * params.seqlen_q *
  // params.h * params.d * sizeof(scalar_type), cuda_stream);
  // cudaMemsetAsync(buffers[10 + buf_offset], 0, params.b * params.seqlen_q *
  // params.h * params.d * sizeof(scalar_type), cuda_stream);
  // cudaMemsetAsync(buffers[11 + buf_offset], 0, params.b * params.seqlen_q *
  // params.h * params.d * sizeof(scalar_type), cuda_stream);
  // cudaMemsetAsync(buffers[12 + buf_offset], 0, params.b * params.seqlen_q *
  // params.h * sizeof(torch::kFloat), cuda_stream);
  dq.fill_(0);
  dk.fill_(0);
  dv.fill_(0);
  dsoftmax_sum.fill_(0);

  int max_seqlen_in_batch_q = params.seqlen_q;
  int max_seqlen_in_batch_k = params.seqlen_k;
  int total_q = params.b * params.seqlen_q;
  int total_k = params.b * params.seqlen_k;
  at::Tensor indices_q =
      cu_seqlens_to_indices(cu_seqlens_q, params.b, params.seqlen_q,
                            scalar_type, max_seqlen_in_batch_q, total_q);
  at::Tensor indices_k;
  if (params.seqlen_q == params.seqlen_k) {
    indices_k = indices_q;
  } else {
    indices_k =
        cu_seqlens_to_indices(cu_seqlens_k, params.b, params.seqlen_k,
                              scalar_type, max_seqlen_in_batch_k, total_k);
  }

  // The unpaded inputs
  auto unpad_do = index_first_axis(do_, indices_q);
  auto unpad_q = index_first_axis(q, indices_q);
  auto unpad_k = index_first_axis(k, indices_k);
  auto unpad_v = index_first_axis(v, indices_k);
  auto unpad_o = index_first_axis(o, indices_q);
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

  launch_params.alibi_slopes_ptr = buf_offset > 0 ? buffers[9] : nullptr;

  launch_params.alibi_slopes_batch_stride = params.alibi_slopes_batch_stride;

  // P = softmax(QK^T)
  launch_params.p_ptr = nullptr;  // no softmax returned always

  // Set the dimensions.
  launch_params.b = params.b;
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
      at::zeros({params.b, params.h, launch_params.seqlen_q_rounded},
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
  launch_params.rng_state = reinterpret_cast<uint64_t*>(buffers[8]);

  launch(launch_params, torch_stream, /*configure=*/false);

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

  cudaEventRecord(xla_wait_torch_event, torch_stream);
  cudaStreamWaitEvent(stream, xla_wait_torch_event);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(custom_call_flash_attention_varlen_backward,
                                "CUDA");

std::vector<xla::XlaOp> BuildFlashAttentionVarlenBackward(
    const xla::XlaOp& dout, const xla::XlaOp& q, const xla::XlaOp& k,
    const xla::XlaOp& v, const xla::XlaOp& out, const xla::XlaOp& softmax_lse,
    const xla::XlaOp& cu_seqlens_q, const xla::XlaOp& cu_seqlens_k,
    const xla::XlaOp& rng_state, const xla::XlaOp& alibi_slopes,
    const FlashAttentionBackwardParams& params,
    const xla::Shape& output_shape) {
  auto builder = q.builder();
  auto opaque = params.ToString();
  std::vector<xla::XlaOp> operands{
      dout, q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state};
  if (alibi_slopes.valid()) {
    operands.push_back(alibi_slopes);
  }
  xla::XlaOp result =
      xla::CustomCall(builder, "custom_call_flash_attention_varlen_backward",
                      std::move(operands), output_shape, opaque);
  return {xla::GetTupleElement(result, 0), xla::GetTupleElement(result, 1),
          xla::GetTupleElement(result, 2), xla::GetTupleElement(result, 3)};
}

}  // namespace

FlashAttentionVarlenBackward::FlashAttentionVarlenBackward(
    const torch::lazy::Value& dout, const torch::lazy::Value& q,
    const torch::lazy::Value& k, const torch::lazy::Value& v,
    const torch::lazy::Value& out, const torch::lazy::Value& softmax_lse,
    const torch::lazy::Value& cu_seqlens_q,
    const torch::lazy::Value& cu_seqlens_k, const torch::lazy::Value& rng_state,
    const FlashAttentionBackwardParams& params, const std::string& params_str)
    : XlaNode(xla_flash_attention_backward,
              {dout, q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k,
               rng_state},
              NodeOutputShape(q, k, v, softmax_lse),
              /*num_outputs=*/4, torch::lazy::MHash(params_str)),
      params_(params),
      params_str_(params_str) {}

FlashAttentionVarlenBackward::FlashAttentionVarlenBackward(
    const torch::lazy::Value& dout, const torch::lazy::Value& q,
    const torch::lazy::Value& k, const torch::lazy::Value& v,
    const torch::lazy::Value& out, const torch::lazy::Value& softmax_lse,
    const torch::lazy::Value& cu_seqlens_q,
    const torch::lazy::Value& cu_seqlens_k, const torch::lazy::Value& rng_state,
    const torch::lazy::Value& alibi_slopes,
    const FlashAttentionBackwardParams& params, const std::string& params_str)
    : XlaNode(xla_flash_attention_backward,
              {dout, q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k,
               rng_state, alibi_slopes},
              NodeOutputShape(q, k, v, softmax_lse),
              /*num_outputs=*/4, torch::lazy::MHash(params_str)),
      params_(params),
      params_str_(params_str) {}

torch::lazy::NodePtr FlashAttentionVarlenBackward::Clone(
    torch::lazy::OpList operands) const {
  if (operands.size() > 9) {
    torch::lazy::MakeNode<FlashAttentionVarlenBackward>(
        operands.at(0), operands.at(1), operands.at(2), operands.at(3),
        operands.at(4), operands.at(5), operands.at(6), operands.at(7),
        operands.at(8), operands.at(9), params_, params_str_);
  } else {
    torch::lazy::MakeNode<FlashAttentionVarlenBackward>(
        operands.at(0), operands.at(1), operands.at(2), operands.at(3),
        operands.at(4), operands.at(5), operands.at(6), operands.at(7),
        operands.at(8), params_, params_str_);
  }
}

XlaOpVector FlashAttentionVarlenBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp dout = loctx->GetOutputOp(operand(0));
  xla::XlaOp q = loctx->GetOutputOp(operand(1));
  xla::XlaOp k = loctx->GetOutputOp(operand(2));
  xla::XlaOp v = loctx->GetOutputOp(operand(3));
  xla::XlaOp out = loctx->GetOutputOp(operand(4));
  xla::XlaOp softmax_lse = loctx->GetOutputOp(operand(5));
  xla::XlaOp cu_seqlens_q = loctx->GetOutputOp(operand(6));
  xla::XlaOp cu_seqlens_k = loctx->GetOutputOp(operand(7));
  xla::XlaOp rng_state = loctx->GetOutputOp(operand(8));
  xla::XlaOp alibi_slopes =
      operands().size() > 9 ? loctx->GetOutputOp(operand(9)) : xla::XlaOp();
  std::vector<xla::XlaOp> result = BuildFlashAttentionVarlenBackward(
      dout, q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state,
      alibi_slopes, params_, xla_shape());

  return ReturnOps({result}, loctx);
}

}  // namespace torch_xla
