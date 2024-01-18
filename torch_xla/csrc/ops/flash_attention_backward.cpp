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

void run_mha_bwd(Flash_bwd_params& params, cudaStream_t stream,
                 const bool configure) {
  FP16_SWITCH(!params.is_bf16, [&] {
    if (params.d <= 32) {
      run_mha_bwd_<elem_type, 32>(params, stream, configure);
    } else if (params.d <= 64) {
      run_mha_bwd_<elem_type, 64>(params, stream, configure);
    } else if (params.d <= 96) {
      run_mha_bwd_<elem_type, 96>(params, stream, configure);
    } else if (params.d <= 128) {
      run_mha_bwd_<elem_type, 128>(params, stream, configure);
    } else if (params.d <= 160) {
      run_mha_bwd_<elem_type, 160>(params, stream, configure);
    } else if (params.d <= 192) {
      run_mha_bwd_<elem_type, 192>(params, stream, configure);
    } else if (params.d <= 224) {
      run_mha_bwd_<elem_type, 224>(params, stream, configure);
    } else if (params.d <= 256) {
      run_mha_bwd_<elem_type, 256>(params, stream, configure);
    }
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

  launch_params.cu_seqlens_q = static_cast<int*>(buffers[6]);
  launch_params.cu_seqlens_k = static_cast<int*>(buffers[7]);

  // P = softmax(QK^T)
  launch_params.p_ptr = nullptr;  // no softmax returned always

  // Softmax sum
  launch_params.softmax_lse_ptr = buffers[5];

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

  launch_params.do_ptr = buffers[0];
  launch_params.do_row_stride = params.do_row_stride;
  launch_params.do_head_stride = params.do_head_stride;
  launch_params.dq_ptr = buffers[8];
  launch_params.dk_ptr = buffers[9];
  launch_params.dv_ptr = buffers[10];
  launch_params.dq_row_stride = params.dq_row_stride;
  launch_params.dk_row_stride = params.dk_row_stride;
  launch_params.dv_row_stride = params.dv_row_stride;
  launch_params.dq_head_stride = params.dq_head_stride;
  launch_params.dk_head_stride = params.dk_head_stride;
  launch_params.dv_head_stride = params.dv_head_stride;

  // bool loop = max_seqlen_k > blocksize_c;
  // TODO: change later, for now set to true for simplicity
  bool loop = true;

  cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream().stream();
  cudaEvent_t torch_wait_xla_event;
  cudaEventCreateWithFlags(&torch_wait_xla_event, cudaEventDisableTiming);
  cudaEvent_t xla_wait_torch_event;
  cudaEventCreateWithFlags(&xla_wait_torch_event, cudaEventDisableTiming);
  cudaEventRecord(torch_wait_xla_event, stream);
  cudaStreamWaitEvent(torch_stream, torch_wait_xla_event);

  auto scalar_type = params.is_bf16 ? torch::kBFloat16 : torch::kFloat16;
  auto opts = torch::TensorOptions().dtype(scalar_type).device(torch::kCUDA);
  at::Tensor dq_accum;
  if (loop) {
    dq_accum =
        torch::empty({launch_params.b, launch_params.h,
                      launch_params.seqlen_q_rounded, launch_params.d_rounded},
                     opts.dtype(at::kFloat));
  }

  at::Tensor dk = torch::from_blob(
      buffers[9], {params.total_k, launch_params.h_k, launch_params.d}, opts);
  at::Tensor dv = torch::from_blob(
      buffers[10], {params.total_k, launch_params.h_k, launch_params.d}, opts);

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
  launch_params.dsoftmax_sum = buffers[11];

  auto launch = &run_mha_bwd;

  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

  // We use a custom RNG that increases the offset by batch_size * nheads * 32.
  int64_t counter_offset = launch_params.b * launch_params.h * 32;

  bool is_dropout = (1.f - launch_params.p_dropout) > 0.0;
  if (is_dropout) {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    launch_params.philox_args = gen->philox_cuda_state(counter_offset);
  }

  launch(launch_params, torch_stream, /*configure=*/false);

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

  cudaEventRecord(xla_wait_torch_event, torch_stream);
  cudaStreamWaitEvent(stream, xla_wait_torch_event);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(custom_call_flash_attention_backward, "CUDA");

std::vector<xla::XlaOp> BuildFlashAttentionBackward(
    const xla::XlaOp& dout, const xla::XlaOp& q, const xla::XlaOp& k,
    const xla::XlaOp& v, const xla::XlaOp& out, const xla::XlaOp& softmax_lse,
    const xla::XlaOp& cu_seqlens_q, const xla::XlaOp& cu_seqlens_k,
    const FlashAttentionBackwardParams& params,
    const xla::Shape& output_shape) {
  auto builder = q.builder();
  auto opaque = params.ToString();
  xla::XlaOp result = xla::CustomCall(
      builder, "custom_call_flash_attention_backward",
      {dout, q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k},
      output_shape, opaque);
  return {xla::GetTupleElement(result, 0), xla::GetTupleElement(result, 1),
          xla::GetTupleElement(result, 2), xla::GetTupleElement(result, 3)};
}

}  // namespace

FlashAttentionBackward::FlashAttentionBackward(
    const torch::lazy::Value& dout, const torch::lazy::Value& q,
    const torch::lazy::Value& k, const torch::lazy::Value& v,
    const torch::lazy::Value& out, const torch::lazy::Value& softmax_lse,
    const torch::lazy::Value& cu_seqlens_q,
    const torch::lazy::Value& cu_seqlens_k,
    const FlashAttentionBackwardParams& params)
    : XlaNode(xla_flash_attention_backward,
              {dout, q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k},
              NodeOutputShape(q, k, v, softmax_lse),
              /*num_outputs=*/4,
              torch::lazy::MHash(params.b, params.h, params.seqlen_q)),
      params_(params) {}

torch::lazy::NodePtr FlashAttentionBackward::Clone(
    torch::lazy::OpList operands) const {
  torch::lazy::MakeNode<FlashAttentionBackward>(
      operands.at(0), operands.at(1), operands.at(2), operands.at(3),
      operands.at(4), operands.at(5), operands.at(6), operands.at(7), params_);
}

XlaOpVector FlashAttentionBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp dout = loctx->GetOutputOp(operand(0));
  xla::XlaOp q = loctx->GetOutputOp(operand(1));
  xla::XlaOp k = loctx->GetOutputOp(operand(2));
  xla::XlaOp v = loctx->GetOutputOp(operand(3));
  xla::XlaOp out = loctx->GetOutputOp(operand(4));
  xla::XlaOp softmax_lse = loctx->GetOutputOp(operand(5));
  xla::XlaOp cu_seqlens_q = loctx->GetOutputOp(operand(6));
  xla::XlaOp cu_seqlens_k = loctx->GetOutputOp(operand(7));

  std::vector<xla::XlaOp> result =
      BuildFlashAttentionBackward(dout, q, k, v, out, softmax_lse, cu_seqlens_q,
                                  cu_seqlens_k, params_, xla_shape());

  return ReturnOps({result}, loctx);
}

}  // namespace torch_xla
