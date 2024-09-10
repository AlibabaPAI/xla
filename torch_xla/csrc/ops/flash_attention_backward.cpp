#include "torch_xla/csrc/ops/flash_attention_backward.h"

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

void run_mha_bwd(Flash_bwd_params& params, cudaStream_t stream) {
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
//  buffers[6] = rng_state
//  buffers[7] = alibi_slopes
//  buffers[8] = dq  // this is output
//  buffers[9] = dk  // this is output
//  buffers[10] = dv  // this is output
//  buffers[11] = softmax_d  // this is output
void custom_call_flash_attention_backward(cudaStream_t stream, void** buffers,
                                          const char* opaque,
                                          size_t opaque_len) {
  std::string opaque_str(opaque, opaque_len);
  TF_VLOG(3) << "custom_call_flash_attention_backward opaque str: "
             << opaque_str;
  FlashAttentionBackwardParams params;
  params.FromString(std::move(opaque_str));
  int buf_offset = params.enable_alibi_slopes;

  if (params.is_causal) { params.window_size_right = 0; }
  if (params.window_size_left >= params.seqlen_k) {
     params.window_size_left = -1;
  }
  if (params.window_size_right >= params.seqlen_k) {
     params.window_size_left = -1;
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int seqlen_q_rounded = round_multiple(params.seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(params.seqlen_k, 128);

  cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream().stream();
  cudaEvent_t torch_wait_xla_event;
  cudaEventCreateWithFlags(&torch_wait_xla_event, cudaEventDisableTiming);
  cudaEvent_t xla_wait_torch_event;
  cudaEventCreateWithFlags(&xla_wait_torch_event, cudaEventDisableTiming);
  cudaEventRecord(torch_wait_xla_event, stream);
  cudaStreamWaitEvent(torch_stream, torch_wait_xla_event);
  auto scalar_type = params.is_bf16 ? torch::kBFloat16 : torch::kFloat16;
  auto opts = torch::TensorOptions().dtype(scalar_type).device(torch::kCUDA);

  at::Tensor q = torch::from_blob(
      buffers[1],
      {params.b, params.seqlen_q, params.h, params.d}, opts);

  at::Tensor dsoftmax_sum = torch::from_blob(
    buffers[10 + buf_offset],
    {params.b, params.h, params.seqlen_q}, opts.dtype(torch::kFloat));
  at::Tensor rounded_dsoftmax_sum = at::empty(
    {params.b, params.h, seqlen_q_rounded}, opts.dtype(torch::kFloat));

  // bool loop = max_seqlen_k > blocksize_c;
  // TODO: change later, for now set to true for simplicity
  bool loop = true;

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  at::Tensor dq_accum;
  if (loop) {
    if (!params.deterministic) {
      dq_accum = torch::empty({
          params.b, seqlen_q_rounded, params.h, params.d_rounded},
        opts.dtype(at::kFloat));
    } else {
      auto dprops = at::cuda::getCurrentDeviceProperties();
      const int nsplits = (dprops->multiProcessorCount +
                           params.b * params.h - 1) /
                          (params.b * params.h);
      dq_accum = torch::zeros({
        nsplits, params.b, seqlen_q_rounded, params.h, params.d_rounded},
        opts.dtype(at::kFloat));
    }
  }

  at::Tensor dq = torch::from_blob(
      buffers[7 + buf_offset],
      {params.b, params.seqlen_q, params.h, params.d}, opts);
  at::Tensor dk = torch::from_blob(
      buffers[8 + buf_offset],
      {params.b, params.seqlen_k, params.h_k, params.d}, opts);
  at::Tensor dv = torch::from_blob(
      buffers[9 + buf_offset ],
      {params.b, params.seqlen_k, params.h_k, params.d}, opts);

  at::Tensor dk_expanded, dv_expanded;

  if (params.h_k != params.h) {  // MQA / GQA
    TF_VLOG(2) << "Running FlashAttention Backward as MQA/GQA";
    dk_expanded =
        torch::empty({params.b, params.seqlen_k, params.h, params.d}, opts);
    dv_expanded =
        torch::empty({params.b, params.seqlen_k, params.h, params.d}, opts);
  } else {
    TF_VLOG(2) << "Running FlashAttention Backward";
    dk_expanded = dk;
    dv_expanded = dv;
  }

  Flash_bwd_params launch_params;

  // Reset the parameters
  memset(&launch_params, 0, sizeof(launch_params));

  launch_params.is_bf16 = params.is_bf16;

  // Set the pointers and strides.
  launch_params.q_ptr = buffers[1];
  launch_params.k_ptr = buffers[2];
  launch_params.v_ptr = buffers[3];
  // All stride are in elements, not bytes.
  launch_params.q_row_stride = params.q_row_stride;
  launch_params.k_row_stride = params.k_row_stride;
  launch_params.v_row_stride = params.v_row_stride;
  launch_params.q_head_stride = params.q_head_stride;
  launch_params.k_head_stride = params.k_head_stride;
  launch_params.v_head_stride = params.v_head_stride;
  launch_params.o_ptr = buffers[4];
  launch_params.o_row_stride = params.o_row_stride;
  launch_params.o_head_stride = params.o_head_stride;

  launch_params.q_batch_stride = dq.stride(0);
  launch_params.k_batch_stride = dk.stride(0);
  launch_params.v_batch_stride = dv.stride(0);
  launch_params.o_batch_stride = dq.stride(0);
  // if (seqlenq_ngroups_swapped) { // TODO
  //       launch_params.q_batch_stride *= seqlen_q;
  //       launch_params.o_batch_stride *= seqlen_q;
  // }

  launch_params.cu_seqlens_q = static_cast<int *>(nullptr);
  launch_params.cu_seqlens_k = static_cast<int *>(nullptr);
  launch_params.seqused_k = static_cast<int *>(nullptr);

  // Softmax sum
  launch_params.softmax_lse_ptr = buffers[5];

  // Set the dimensions.
  launch_params.b = params.b;
  launch_params.h = params.h;
  launch_params.h_k = params.h_k;
  launch_params.h_h_k_ratio = params.h_h_k_ratio;
  launch_params.seqlen_q = params.seqlen_q;
  launch_params.seqlen_k = params.seqlen_k;
  launch_params.seqlen_q_rounded = seqlen_q_rounded;
  launch_params.seqlen_k_rounded = seqlen_k_rounded;
  launch_params.d = params.d;
  launch_params.d_rounded = params.d_rounded;

  // Set the different scale values.
  launch_params.scale_softmax = params.scale_softmax;
  launch_params.scale_softmax_log2 = params.scale_softmax_log2;

  launch_params.p_dropout = params.p_dropout;
  launch_params.p_dropout_in_uint8_t = params.p_dropout_in_uint8_t;
  launch_params.rp_dropout = params.rp_dropout;
  launch_params.scale_softmax_rp_dropout = params.scale_softmax_rp_dropout;

  // Causal is the special case where window_size_right == 0 and window_size_left < 0.
  // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
  launch_params.is_causal = params.window_size_left < 0 && params.window_size_right == 0;

  if (params.window_size_left < 0 && params.window_size_right >= 0) {
    params.window_size_left = params.seqlen_k;
  }
  if (params.window_size_left >= 0 && params.window_size_right < 0) {
    params.window_size_right = params.seqlen_k;
  }

  launch_params.window_size_left = params.window_size_left;
  launch_params.window_size_right = params.window_size_right;

  launch_params.is_seqlens_k_cumulative = true;

  // P = softmax(QK^T)
  launch_params.p_ptr = nullptr;  // no softmax returned always

  launch_params.do_ptr = buffers[0];
  launch_params.do_row_stride = params.do_row_stride;
  launch_params.do_head_stride = params.do_head_stride;
  launch_params.dq_ptr = buffers[7 + buf_offset];
  launch_params.dq_row_stride = params.dq_row_stride;
  launch_params.dq_head_stride = params.dq_head_stride;
  launch_params.dk_ptr = dk_expanded.data_ptr();
  launch_params.dv_ptr = dv_expanded.data_ptr();
  launch_params.dk_row_stride = dk_expanded.stride(-3);
  launch_params.dv_row_stride = dv_expanded.stride(-3);
  launch_params.dk_head_stride = dk_expanded.stride(-2);
  launch_params.dv_head_stride = dv_expanded.stride(-2);

  launch_params.do_batch_stride = dq.stride(0);
  launch_params.dq_batch_stride = dq.stride(0);
  launch_params.dk_batch_stride = dk.stride(0);
  launch_params.dv_batch_stride = dv.stride(0);

  launch_params.dq_accum_ptr = loop ? dq_accum.data_ptr() : nullptr;
  launch_params.dk_accum_ptr = nullptr;
  launch_params.dv_accum_ptr = nullptr;

  launch_params.dsoftmax_sum = rounded_dsoftmax_sum.data_ptr();

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
  launch_params.rng_state = reinterpret_cast<uint64_t*>(buffers[6]);
  at::Tensor rng_state = torch::from_blob(
    buffers[6],
    {2}, opts.dtype(torch::kInt));
  launch_params.alibi_slopes_ptr = buf_offset > 0 ? buffers[7] : nullptr;
  launch_params.alibi_slopes_batch_stride = params.alibi_slopes_batch_stride;

  launch(launch_params, torch_stream);

  // For MQA/GQA we need to sum dK and dV across the groups
  if (launch_params.h_k != launch_params.h) {
    at::sum_out(dk,
                at::reshape(dk_expanded, {launch_params.b, params.seqlen_k, launch_params.h_k,
                                          launch_params.h / launch_params.h_k,
                                          launch_params.d}),
                {3});
    at::sum_out(dv,
                at::reshape(dv_expanded, {launch_params.b, params.seqlen_k, launch_params.h_k,
                                          launch_params.h / launch_params.h_k,
                                          launch_params.d}),
                {3});
  }

  dsoftmax_sum.copy_(rounded_dsoftmax_sum.slice(2, 0, params.seqlen_q));

  cudaEventRecord(xla_wait_torch_event, torch_stream);
  cudaStreamWaitEvent(stream, xla_wait_torch_event);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(custom_call_flash_attention_backward, "CUDA");

std::vector<xla::XlaOp> BuildFlashAttentionBackward(
    const xla::XlaOp& dout, const xla::XlaOp& q, const xla::XlaOp& k,
    const xla::XlaOp& v, const xla::XlaOp& out, const xla::XlaOp& softmax_lse,
    const xla::XlaOp& rng_state, const xla::XlaOp& alibi_slopes,
    const FlashAttentionBackwardParams& params,
    const xla::Shape& output_shape) {
  auto builder = q.builder();
  auto opaque = params.ToString();
  std::vector<xla::XlaOp> operands{
      dout, q, k, v, out, softmax_lse, rng_state};
  if (alibi_slopes.valid()) {
    operands.push_back(alibi_slopes);
  }
  xla::XlaOp result =
      xla::CustomCall(builder, "custom_call_flash_attention_backward",
                      std::move(operands), output_shape, opaque);
  return {xla::GetTupleElement(result, 0), xla::GetTupleElement(result, 1),
          xla::GetTupleElement(result, 2), xla::GetTupleElement(result, 3)};
}

}  // namespace

FlashAttentionBackward::FlashAttentionBackward(
    const torch::lazy::Value& dout, const torch::lazy::Value& q,
    const torch::lazy::Value& k, const torch::lazy::Value& v,
    const torch::lazy::Value& out, const torch::lazy::Value& softmax_lse,
    const torch::lazy::Value& rng_state,
    const FlashAttentionBackwardParams& params,
    const std::string& params_str)
    : XlaNode(xla_flash_attention_backward,
              {dout, q, k, v, out, softmax_lse,
               rng_state},
              NodeOutputShape(q, k, v, softmax_lse),
              /*num_outputs=*/4,
              torch::lazy::MHash(params_str)),
      params_(params),
      params_str_(params_str) {}

FlashAttentionBackward::FlashAttentionBackward(
    const torch::lazy::Value& dout, const torch::lazy::Value& q,
    const torch::lazy::Value& k, const torch::lazy::Value& v,
    const torch::lazy::Value& out, const torch::lazy::Value& softmax_lse,
     const torch::lazy::Value& rng_state,
    const torch::lazy::Value& alibi_slopes,
    const FlashAttentionBackwardParams& params,
    const std::string& params_str)
    : XlaNode(xla_flash_attention_backward,
              {dout, q, k, v, out, softmax_lse,
               rng_state, alibi_slopes},
              NodeOutputShape(q, k, v, softmax_lse),
              /*num_outputs=*/4,
              torch::lazy::MHash(params_str)),
      params_(params),
      params_str_(params_str) {}

torch::lazy::NodePtr FlashAttentionBackward::Clone(
    torch::lazy::OpList operands) const {
  if (operands.size() > 7) {
    torch::lazy::MakeNode<FlashAttentionBackward>(
        operands.at(0), operands.at(1), operands.at(2), operands.at(3),
        operands.at(4), operands.at(5), operands.at(6), operands.at(7),
        params_, params_str_);
  } else {
    torch::lazy::MakeNode<FlashAttentionBackward>(
        operands.at(0), operands.at(1), operands.at(2), operands.at(3),
        operands.at(4), operands.at(5), operands.at(6), params_, params_str_);
  }
}

XlaOpVector FlashAttentionBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp dout = loctx->GetOutputOp(operand(0));
  xla::XlaOp q = loctx->GetOutputOp(operand(1));
  xla::XlaOp k = loctx->GetOutputOp(operand(2));
  xla::XlaOp v = loctx->GetOutputOp(operand(3));
  xla::XlaOp out = loctx->GetOutputOp(operand(4));
  xla::XlaOp softmax_lse = loctx->GetOutputOp(operand(5));
  xla::XlaOp rng_state = loctx->GetOutputOp(operand(6));
  xla::XlaOp alibi_slopes =
      operands().size() > 7 ? loctx->GetOutputOp(operand(7)) : xla::XlaOp();
  std::vector<xla::XlaOp> result = BuildFlashAttentionBackward(
      dout, q, k, v, out, softmax_lse, rng_state,
      alibi_slopes, params_, xla_shape());

  return ReturnOps({result}, loctx);
}

}  // namespace torch_xla
