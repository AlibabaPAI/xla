#include "torch_xla/csrc/ops/flash_attention_forward.h"

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

xla::Shape NodeOutputShape(int batch_size, int num_heads, int seqlen_q,
                           const torch::lazy::Value& q) {
  xla::Shape softmax_lse_shape = xla::ShapeUtil::MakeShape(
      xla::PrimitiveType::F32, {batch_size, num_heads, seqlen_q});
  xla::Shape out_shape = GetXlaShape(q);
  xla::Shape rng_state_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::S64, {2});
  return xla::ShapeUtil::MakeTupleShape(
      {softmax_lse_shape, out_shape, rng_state_shape});
}

// Layout of `buffers` listed above:
//  buffers[0] = q
//  buffers[1] = k
//  buffers[2] = v
//  buffers[3] = cu_seqlens_q
//  buffers[4] = cu_seqlens_k
//  buffers[5] = alibi_slopes
//  buffers[6] = softmax_lse  // this is output
//  buffers[7] = out_for_output // this is output
//  buffers[8] = rng_state // this is output
void custom_call_flash_attention_forward(cudaStream_t stream, void** buffers,
                                         const char* opaque,
                                         size_t opaque_len) {
  std::string opaque_str(opaque, opaque_len);
  TF_VLOG(3) << "custom_call_flash_attention_forward opaque str: "
             << opaque_str;
  FlashAttentionForwardParams params;
  params.FromString(std::move(opaque_str));
  int buf_offset = params.enable_alibi_slopes;

  auto dprops = at::cuda::getCurrentDeviceProperties();

  cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream().stream();
  Flash_fwd_params launch_params;

  // Reset the parameters
  memset(&launch_params, 0, sizeof(launch_params));

  launch_params.is_bf16 = params.is_bf16;

  // Set the pointers and strides.
  launch_params.q_ptr = buffers[0];
  launch_params.k_ptr = buffers[1];
  launch_params.v_ptr = buffers[2];
  // All stride are in elements, not bytes.
  launch_params.q_row_stride = params.q_row_stride;
  launch_params.k_row_stride = params.k_row_stride;
  launch_params.v_row_stride = params.v_row_stride;
  launch_params.q_head_stride = params.q_head_stride;
  launch_params.k_head_stride = params.k_head_stride;
  launch_params.v_head_stride = params.v_head_stride;
  launch_params.o_ptr = buffers[6 + buf_offset];
  launch_params.o_row_stride = params.o_row_stride;
  launch_params.o_head_stride = params.o_head_stride;

  launch_params.cu_seqlens_q = static_cast<int*>(buffers[3]);
  launch_params.cu_seqlens_k = static_cast<int*>(buffers[4]);
  launch_params.alibi_slopes_ptr = buf_offset > 0 ? buffers[5] : nullptr;
  launch_params.alibi_slopes_batch_stride = params.alibi_slopes_batch_stride;

  // P = softmax(QK^T)
  launch_params.p_ptr = nullptr;  // no softmax returned always

  // Softmax sum
  launch_params.softmax_lse_ptr = buffers[5 + buf_offset];

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

  launch_params.is_seqlens_k_cumulative = params.is_seqlens_k_cumulative;

  // set params splitkv
  launch_params.num_splits = params.num_splits;

  int64_t counter_offset = params.b * params.h * 32;
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  // Forward kernel will populate memory with the seed and offset.
  launch_params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
  buffers[7 + buf_offset] = launch_params.rng_state;

  if ((1.f - launch_params.p_dropout) > 0.0) {
    // number of times random will be generated per thread, to offset philox
    // counter in thc random state We use a custom RNG that increases the offset
    // by batch_size * nheads * 32.
    int64_t counter_offset = launch_params.b * launch_params.h * 32;
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    launch_params.philox_args = gen->philox_cuda_state(counter_offset);
  }

  TF_VLOG(2) << "Running FlashAttention Forward.";
  cudaEvent_t torch_wait_xla_event;
  cudaEventCreateWithFlags(&torch_wait_xla_event, cudaEventDisableTiming);
  cudaEvent_t xla_wait_torch_event;
  cudaEventCreateWithFlags(&xla_wait_torch_event, cudaEventDisableTiming);
  cudaEventRecord(torch_wait_xla_event, stream);
  cudaStreamWaitEvent(torch_stream, torch_wait_xla_event);
  FP16_SWITCH(!launch_params.is_bf16, [&] {
    HEADDIM_SWITCH(launch_params.d, [&] {
      // TODO(wenting.swt): support split_kv
      run_mha_fwd_<elem_type, kHeadDim>(launch_params, torch_stream);
    });
  });
  cudaEventRecord(xla_wait_torch_event, torch_stream);
  cudaStreamWaitEvent(stream, xla_wait_torch_event);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(custom_call_flash_attention_forward, "CUDA");

std::vector<xla::XlaOp> BuildFlashAttentionForward(
    const xla::XlaOp& q, const xla::XlaOp& k, const xla::XlaOp& v,
    const xla::XlaOp& cu_seqlens_q, const xla::XlaOp& cu_seqlens_k,
    const xla::XlaOp& alibi_slopes, const FlashAttentionForwardParams& params,
    const xla::Shape& output_shape) {
  auto builder = q.builder();
  auto opaque = params.ToString();
  std::vector<xla::XlaOp> operands{q, k, v, cu_seqlens_q, cu_seqlens_k};
  if (alibi_slopes.valid()) {
    operands.push_back(alibi_slopes);
  }
  xla::XlaOp result =
      xla::CustomCall(builder, "custom_call_flash_attention_forward",
                      std::move(operands), output_shape, opaque);
  return {xla::GetTupleElement(result, 0), xla::GetTupleElement(result, 1),
          xla::GetTupleElement(result, 2)};
}

}  // namespace

FlashAttentionForward::FlashAttentionForward(
    const torch::lazy::Value& q, const torch::lazy::Value& k,
    const torch::lazy::Value& v, const torch::lazy::Value& cu_seqlens_q,
    const torch::lazy::Value& cu_seqlens_k,
    const FlashAttentionForwardParams& params)
    : XlaNode(xla_flash_attention_forward,
              {q, k, v, cu_seqlens_q, cu_seqlens_k},
              NodeOutputShape(params.b, params.h, params.seqlen_q, q),
              /*num_outputs=*/3,
              torch::lazy::MHash(params.b, params.h, params.seqlen_q)),
      params_(params) {}

FlashAttentionForward::FlashAttentionForward(
    const torch::lazy::Value& q, const torch::lazy::Value& k,
    const torch::lazy::Value& v, const torch::lazy::Value& cu_seqlens_q,
    const torch::lazy::Value& cu_seqlens_k,
    const torch::lazy::Value& alibi_slopes,
    const FlashAttentionForwardParams& params)
    : XlaNode(xla_flash_attention_forward,
              {q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes},
              NodeOutputShape(params.b, params.h, params.seqlen_q, q),
              /*num_outputs=*/3,
              torch::lazy::MHash(params.b, params.h, params.seqlen_q)),
      params_(params) {}

torch::lazy::NodePtr FlashAttentionForward::Clone(
    torch::lazy::OpList operands) const {
  if (operands.size() > 5) {
    torch::lazy::MakeNode<FlashAttentionForward>(
        operands.at(0), operands.at(1), operands.at(2), operands.at(3),
        operands.at(4), operands.at(5), params_);
  } else {
    torch::lazy::MakeNode<FlashAttentionForward>(operands.at(0), operands.at(1),
                                                 operands.at(2), operands.at(3),
                                                 operands.at(4), params_);
  }
}

XlaOpVector FlashAttentionForward::Lower(LoweringContext* loctx) const {
  xla::XlaOp q = loctx->GetOutputOp(operand(0));
  xla::XlaOp k = loctx->GetOutputOp(operand(1));
  xla::XlaOp v = loctx->GetOutputOp(operand(2));
  xla::XlaOp cu_seqlens_q = loctx->GetOutputOp(operand(3));
  xla::XlaOp cu_seqlens_k = loctx->GetOutputOp(operand(4));
  xla::XlaOp alibi_slopes =
      operands().size() > 5 ? loctx->GetOutputOp(operand(5)) : xla::XlaOp();
  std::vector<xla::XlaOp> result = BuildFlashAttentionForward(
      q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, params_, xla_shape());
  return ReturnOps({result}, loctx);
}

}  // namespace torch_xla
