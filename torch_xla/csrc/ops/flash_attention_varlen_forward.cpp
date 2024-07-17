#include "torch_xla/csrc/ops/flash_attention_varlen_forward.h"

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
      xla::PrimitiveType::F32, {batch_size, num_heads, seqlen_q}); // seqlen_q: padding
  xla::Shape out_shape = GetXlaShape(q);
  xla::Shape rng_state_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::U64, {2});
  xla::Shape cu_seqlens_shape = xla::ShapeUtil::MakeShape(
      xla::PrimitiveType::S32, {batch_size + 1});
  return xla::ShapeUtil::MakeTupleShape(
      {softmax_lse_shape, out_shape, rng_state_shape, cu_seqlens_shape, cu_seqlens_shape});
}

// Layout of `buffers` listed above:
//  buffers[0] = q
//  buffers[1] = k
//  buffers[2] = v
//  buffers[3] = attention_mask
//  buffers[4] = alibi_slopes
//  buffers[5] = softmax_lse  // this is output
//  buffers[6] = out_for_output // this is output
//  buffers[7] = rng_state // this is output
//  buffers[8] = cu_seqlen_q // this is output
//  buffers[9] = cu_seqlen_k // this is output
void custom_call_flash_attention_varlen_forward(cudaStream_t stream, void** buffers,
                                         const char* opaque,
                                         size_t opaque_len) {
  std::string opaque_str(opaque, opaque_len);
  TF_VLOG(3) << "custom_call_flash_attention_varlen_forward opaque str: "
             << opaque_str;
  FlashAttentionForwardParams params;
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

  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  at::Tensor attention_mask = torch::from_blob(
      buffers[3],
      {params.b, params.seqlen_k}, opts);
  at::Tensor cu_seqlens_q = torch::from_blob(
      buffers[7 + buf_offset],
      {params.b + 1}, opts);
  at::Tensor cu_seqlens_k = torch::from_blob(
      buffers[8 + buf_offset],
      {params.b + 1}, opts);
  at::Tensor q = torch::from_blob(
      buffers[0],
      {params.b * params.seqlen_q, params.h, params.d}, opts.dtype(scalar_type));
  at::Tensor k = torch::from_blob(
      buffers[1],
      {params.b * params.seqlen_k, params.h_k, params.d}, opts.dtype(scalar_type));
  at::Tensor v = torch::from_blob(
      buffers[2],
      {params.b * params.seqlen_k, params.h_k, params.d}, opts.dtype(scalar_type));
  int max_seqlen_in_batch_k = params.seqlen_k;
  int total_k = params.b * params.seqlen_k;
  at::Tensor indices_k = mask_to_indices(attention_mask, max_seqlen_in_batch_k, total_k, cu_seqlens_k);
  auto unpad_k = index_first_axis(k, indices_k);
  auto unpad_v = index_first_axis(v, indices_k);

  int max_seqlen_in_batch_q = max_seqlen_in_batch_k;
  int total_q = total_k;
  at::Tensor indices_q;

  if (params.seqlen_q == params.seqlen_k) {
    cu_seqlens_q.copy_(cu_seqlens_k);
    indices_q = indices_k;
  } else if (params.seqlen_q == 1) {
    max_seqlen_in_batch_q = 1;
    indices_q = cu_seqlens_q.slice(/*dim=*/0, /*start=*/0, /*end=*/params.b);
    total_q = params.b;
  } else {
    at::Tensor attention_mask_slice = attention_mask.slice(/*dim=*/1, /*start=*/-params.seqlen_q, /*end=*/torch::indexing::None);
    indices_q = mask_to_indices(attention_mask_slice, max_seqlen_in_batch_q, total_q, cu_seqlens_q);
  }
  
  at::Tensor unpad_q = index_first_axis(q, indices_q);

  at::Tensor unpad_output = torch::zeros(
    {total_q, params.h * params.d}, opts.dtype(scalar_type));
  at::Tensor unpad_softmax_lse = torch::zeros(
    {params.b, params.h, max_seqlen_in_batch_q}, opts.dtype(torch::kFloat));

  if (max_seqlen_in_batch_q == 1) { params.is_causal = false; }
  if (params.is_causal) { params.window_size_right = 0; }

  if (params.window_size_left >= max_seqlen_in_batch_k) {
     params.window_size_left = -1;
  }
  if (params.window_size_right >= max_seqlen_in_batch_k) {
     params.window_size_right = -1;
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  auto dprops = at::cuda::getCurrentDeviceProperties();

  Flash_fwd_params launch_params;

  // Reset the parameters
  memset(&launch_params, 0, sizeof(launch_params));

  launch_params.is_bf16 = params.is_bf16;

  // Set the pointers and strides.
  launch_params.q_ptr = unpad_q.data_ptr();
  launch_params.k_ptr = unpad_k.data_ptr();
  launch_params.v_ptr = unpad_v.data_ptr();
  // All stride are in elements, not bytes.
  launch_params.q_row_stride = params.q_row_stride;
  launch_params.k_row_stride = params.k_row_stride;
  launch_params.v_row_stride = params.v_row_stride;
  launch_params.q_head_stride = params.q_head_stride;
  launch_params.k_head_stride = params.k_head_stride;
  launch_params.v_head_stride = params.v_head_stride;
  launch_params.o_ptr = unpad_output.data_ptr();
  launch_params.o_row_stride = params.o_row_stride;
  launch_params.o_head_stride = params.o_head_stride;

  launch_params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q.data_ptr());
  launch_params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k.data_ptr());
  launch_params.seqused_k = static_cast<int*>(nullptr);

  // P = softmax(QK^T)
  launch_params.p_ptr = nullptr;  // no softmax returned always

  // Softmax sum
  launch_params.softmax_lse_ptr = unpad_softmax_lse.data_ptr();

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

  launch_params.is_causal = params.window_size_left < 0 && params.window_size_right == 0;

  if (params.window_size_left < 0 && params.window_size_right >= 0) {
    params.window_size_left = max_seqlen_in_batch_k;
  }
  if (params.window_size_left >= 0 && params.window_size_right < 0) {
    params.window_size_right = max_seqlen_in_batch_k;
  }

  launch_params.window_size_left = params.window_size_left;
  launch_params.window_size_right = params.window_size_right;

  launch_params.is_seqlens_k_cumulative = params.is_seqlens_k_cumulative;

  launch_params.alibi_slopes_ptr = buf_offset > 0 ? buffers[4] : nullptr;
  launch_params.alibi_slopes_batch_stride = params.alibi_slopes_batch_stride;


  // set params splitkv
  launch_params.num_splits = params.num_splits;

  int64_t counter_offset = params.b * params.h * 32;
  auto options =
      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
  auto rng_state = torch::from_blob(
      buffers[6 + buf_offset],
      {2}, options.dtype(torch::kInt64));
  // Forward kernel will populate memory with the seed and offset.
  launch_params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
  rng_state.fill_(0);

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

  FP16_SWITCH(!launch_params.is_bf16, [&] {
    HEADDIM_SWITCH(launch_params.d, [&] {
      // TODO(wenting.swt): support split_kv
      run_mha_fwd_<elem_type, kHeadDim>(launch_params, torch_stream);
    });
  });

  at::Tensor softmax_lse = torch::from_blob(
    buffers[4 + buf_offset],
    {params.b, params.h, params.seqlen_q},
    opts.dtype(torch::kFloat)
  );
  softmax_lse.slice(2, 0, max_seqlen_in_batch_q).copy_(unpad_softmax_lse.slice(2, 0, max_seqlen_in_batch_q));

  at::Tensor o_output = torch::from_blob(
      buffers[5 + buf_offset],
      {params.b * params.seqlen_q, params.h * params.d}, opts.dtype(scalar_type));
  torch::Tensor repeated_indices_q = indices_q.unsqueeze(1).expand(
    {indices_q.size(0), params.h * params.d});
  o_output.scatter_(0, repeated_indices_q, unpad_output);

  // TODO(wenting.swt): we should pad and unpad q,k,v when head_size_og % 8 != 0
  // sync with cudaEvent
  cudaEventRecord(xla_wait_torch_event, torch_stream);
  cudaStreamWaitEvent(stream, xla_wait_torch_event);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(custom_call_flash_attention_varlen_forward, "CUDA");

std::vector<xla::XlaOp> BuildFlashAttentionVarlenForward(
    const xla::XlaOp& q, const xla::XlaOp& k, const xla::XlaOp& v,
    const xla::XlaOp& attention_mask,
    const xla::XlaOp& alibi_slopes, const FlashAttentionForwardParams& params,
    const xla::Shape& output_shape) {
  auto builder = q.builder();
  auto opaque = params.ToString();
  std::vector<xla::XlaOp> operands{q, k, v, attention_mask};
  if (alibi_slopes.valid()) {
    operands.push_back(alibi_slopes);
  }
  xla::XlaOp result =
      xla::CustomCall(builder, "custom_call_flash_attention_varlen_forward",
                      std::move(operands), output_shape, opaque);
  return {/*softmax_lse*/xla::GetTupleElement(result, 0),
          /*output*/xla::GetTupleElement(result, 1),
          /*rng_state*/xla::GetTupleElement(result, 2),
          /*cu_seqlen_q*/xla::GetTupleElement(result, 3),
          /*cu_seqlen_k*/xla::GetTupleElement(result, 4)};
}

}  // namespace

FlashAttentionVarlenForward::FlashAttentionVarlenForward(
    const torch::lazy::Value& q, const torch::lazy::Value& k,
    const torch::lazy::Value& v, const torch::lazy::Value& attention_mask,
    const FlashAttentionForwardParams& params,
    const std::string& params_str)
    : XlaNode(xla_flash_attention_forward,
              {q, k, v, attention_mask},
              NodeOutputShape(params.b, params.h, params.seqlen_q, q),
              /*num_outputs=*/5,
              torch::lazy::MHash(params_str)),
      params_(params),
      params_str_(params_str) {}

FlashAttentionVarlenForward::FlashAttentionVarlenForward(
    const torch::lazy::Value& q, const torch::lazy::Value& k,
    const torch::lazy::Value& v, const torch::lazy::Value& attention_mask,
    const torch::lazy::Value& alibi_slopes,
    const FlashAttentionForwardParams& params,
    const std::string& params_str)
    : XlaNode(xla_flash_attention_forward,
              {q, k, v, attention_mask, alibi_slopes},
              NodeOutputShape(params.b, params.h, params.seqlen_q, q),
              /*num_outputs=*/5,
              torch::lazy::MHash(params_str)),
      params_(params),
      params_str_(params_str) {}

torch::lazy::NodePtr FlashAttentionVarlenForward::Clone(
    torch::lazy::OpList operands) const {
  if (operands.size() > 4){
    torch::lazy::MakeNode<FlashAttentionVarlenForward>(operands.at(0), operands.at(1),
                                                      operands.at(2), operands.at(3),
                                                      operands.at(4), params_, params_str_);
  } else {
    torch::lazy::MakeNode<FlashAttentionVarlenForward>(operands.at(0), operands.at(1),
                                                      operands.at(2), operands.at(3),
                                                      params_, params_str_);
  }
}

XlaOpVector FlashAttentionVarlenForward::Lower(LoweringContext* loctx) const {
  xla::XlaOp q = loctx->GetOutputOp(operand(0));
  xla::XlaOp k = loctx->GetOutputOp(operand(1));
  xla::XlaOp v = loctx->GetOutputOp(operand(2));
  xla::XlaOp attention_mask = loctx->GetOutputOp(operand(3));
  xla::XlaOp alibi_slopes =
      operands().size() > 4 ? loctx->GetOutputOp(operand(4)) : xla::XlaOp();
  std::vector<xla::XlaOp> result = BuildFlashAttentionVarlenForward(
      q, k, v, attention_mask, alibi_slopes, params_, xla_shape());
  return ReturnOps({result}, loctx);
}

}  // namespace torch_xla
