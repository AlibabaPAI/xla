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
      xla::PrimitiveType::F32, {batch_size, num_heads, seqlen_q}); // seqlen_q: padding
  xla::Shape out_shape = GetXlaShape(q);
  xla::Shape rng_state_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::U64, {2});
  return xla::ShapeUtil::MakeTupleShape(
      {softmax_lse_shape, out_shape, rng_state_shape});
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

// Layout of `buffers` listed above:
//  buffers[0] = q
//  buffers[1] = k
//  buffers[2] = v
//  buffers[3] = alibi_slopes
//  buffers[4] = softmax_lse  // this is output
//  buffers[5] = out_for_output // this is output
//  buffers[6] = rng_state // this is output
void custom_call_flash_attention_forward(cudaStream_t stream, void** buffers,
                                         const char* opaque,
                                         size_t opaque_len) {
  std::string opaque_str(opaque, opaque_len);
  TF_VLOG(3) << "custom_call_flash_attention_forward opaque str: "
             << opaque_str;
  FlashAttentionForwardParams params;
  params.FromString(std::move(opaque_str));
  int buf_offset = params.enable_alibi_slopes;

  if (params.window_size_left >= params.seqlen_k) {
     params.window_size_left = -1;
  }
  if (params.window_size_right >= params.seqlen_k) {
     params.window_size_left = -1;
  }

  if (params.seqlen_q == 1) { params.is_causal = false; }
  if (params.is_causal) { params.window_size_right = 0; }

  Flash_fwd_params launch_params;

  // Reset the parameters
  memset(&launch_params, 0, sizeof(launch_params));

  cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream().stream();
  cudaEvent_t torch_wait_xla_event;
  cudaEventCreateWithFlags(&torch_wait_xla_event, cudaEventDisableTiming);
  cudaEvent_t xla_wait_torch_event;
  cudaEventCreateWithFlags(&xla_wait_torch_event, cudaEventDisableTiming);
  cudaEventRecord(torch_wait_xla_event, stream);
  cudaStreamWaitEvent(torch_stream, torch_wait_xla_event);

  auto scalar_type = params.is_bf16 ? torch::kBFloat16 : torch::kFloat16;
  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  at::Tensor q = torch::from_blob(
      buffers[0],
      {params.b, params.seqlen_q, params.h, params.d}, opts.dtype(scalar_type));
  at::Tensor k = torch::from_blob(
      buffers[1],
      {params.b, params.seqlen_k, params.h_k, params.d}, opts.dtype(scalar_type));
  at::Tensor v = torch::from_blob(
      buffers[2],
      {params.b, params.seqlen_k, params.h_k, params.d}, opts.dtype(scalar_type));
  at::Tensor out = torch::from_blob(
      buffers[4 + buf_offset],
      {params.b, params.seqlen_q, params.h, params.d}, opts.dtype(scalar_type));

  const int seqlenq_ngroups_swapped = params.seqlen_q == 1 && params.h > params.h_k && params.window_size_left < 0 && params.window_size_right < 0 && 1 - params.p_dropout == 0.f && params.h % 8 == 0 && !params.enable_alibi_slopes;
  if (seqlenq_ngroups_swapped) {
      const int ngroups = params.h / params.h_k;
      q = q.reshape({params.b, params.h_k, ngroups, params.h}).transpose(1, 2);
      params.seqlen_q = ngroups;
      params.h = params.h_k;
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  launch_params.is_bf16 = params.is_bf16;

  // Set the pointers and strides.
  launch_params.q_ptr = q.data_ptr();
  launch_params.k_ptr = k.data_ptr();
  launch_params.v_ptr = v.data_ptr();
  // All stride are in elements, not bytes.
  launch_params.q_row_stride = params.q_row_stride;
  launch_params.k_row_stride = params.k_row_stride;
  launch_params.v_row_stride = params.v_row_stride;
  launch_params.q_head_stride = params.q_head_stride;
  launch_params.k_head_stride = params.k_head_stride;
  launch_params.v_head_stride = params.v_head_stride;
  launch_params.o_ptr = out.data_ptr();
  launch_params.o_row_stride = params.o_row_stride;
  launch_params.o_head_stride = params.o_head_stride;

  launch_params.q_batch_stride = q.stride(0);
  launch_params.k_batch_stride = k.stride(0);
  launch_params.v_batch_stride = v.stride(0);
  launch_params.o_batch_stride = out.stride(0);

  launch_params.cu_seqlens_q = static_cast<int *>(nullptr);
  launch_params.cu_seqlens_k = static_cast<int *>(nullptr);
  launch_params.seqused_k = static_cast<int *>(nullptr);

  // P = softmax(QK^T)
  launch_params.p_ptr = nullptr;  // no softmax returned always

  launch_params.softmax_lse_ptr = buffers[3 + buf_offset];

  // Set the dimensions.
  launch_params.b = params.b;
  launch_params.h = params.h;
  launch_params.h_k = params.h_k;
  launch_params.h_h_k_ratio = params.h_h_k_ratio;
  launch_params.seqlen_q = params.seqlen_q;
  launch_params.seqlen_k = params.seqlen_k;
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  launch_params.seqlen_q_rounded = round_multiple(params.seqlen_q, 128);
  launch_params.seqlen_k_rounded = round_multiple(params.seqlen_k, 128);
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
    params.window_size_left = params.seqlen_k;
  }
  if (params.window_size_left >= 0 && params.window_size_right < 0) {
    params.window_size_right = params.seqlen_k;
  }

  launch_params.window_size_left = params.window_size_left;
  launch_params.window_size_right = params.window_size_right;

  launch_params.is_seqlens_k_cumulative = params.is_seqlens_k_cumulative;
  launch_params.page_block_size = 1;

  // set params splitkv
  const int block_n = params.d <= 64 ? 256 : (params.d <= 128 ? 128 : 64);
  const int num_n_blocks = (params.seqlen_k + block_n - 1) / block_n;
  // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
  // In any case we don't expect seqlen_q to be larger than 64 for inference.
  const int num_m_blocks = (params.seqlen_q + 64 - 1) / 64;
  launch_params.num_splits = 0;
  if (1 - params.p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
      if (launch_params.num_splits < 1) {
          auto dprops = at::cuda::getCurrentDeviceProperties();
          launch_params.num_splits = num_splits_heuristic(params.b * params.h * num_m_blocks, dprops->multiProcessorCount, num_n_blocks, 128);
      }
  }

  int64_t counter_offset = params.b * params.h * 32;

  auto rng_state = torch::from_blob(
      buffers[5 + buf_offset],
      {2}, opts.dtype(torch::kInt64));
  // Forward kernel will populate memory with the seed and offset.
  launch_params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
//   buffers[5 + buf_offset] = launch_params.rng_state;
//   cudaMemsetAsync(launch_params.rng_state, 0, 2 * sizeof(uint64_t));
  rng_state.fill_(0);

  if ((1.f - launch_params.p_dropout) > 0.0) {
    // number of times random will be generated per thread, to offset philox
    // counter in thc random state We use a custom RNG that increases the offset
    // by batch_size * nheads * 32.
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    launch_params.philox_args = gen->philox_cuda_state(counter_offset);
  }

  launch_params.alibi_slopes_ptr = buf_offset > 0 ? buffers[3] : nullptr;
  launch_params.alibi_slopes_batch_stride = params.alibi_slopes_batch_stride;

  TF_VLOG(2) << "Running FlashAttention Forward.";

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
    const xla::XlaOp& alibi_slopes, const FlashAttentionForwardParams& params,
    const xla::Shape& output_shape) {
  auto builder = q.builder();
  auto opaque = params.ToString();
  std::vector<xla::XlaOp> operands{q, k, v};
  if (alibi_slopes.valid()) {
    operands.push_back(alibi_slopes);
  }
  xla::XlaOp result =
      xla::CustomCall(builder, "custom_call_flash_attention_forward",
                      std::move(operands), output_shape, opaque);
  return {/*softmax_lse*/xla::GetTupleElement(result, 0),
          /*output*/xla::GetTupleElement(result, 1),
          /*rng_state*/xla::GetTupleElement(result, 2)};
}

}  // namespace

FlashAttentionForward::FlashAttentionForward(
    const torch::lazy::Value& q, const torch::lazy::Value& k,
    const torch::lazy::Value& v,
    const FlashAttentionForwardParams& params,
    const std::string& params_str)
    : XlaNode(xla_flash_attention_forward,
              {q, k, v},
              NodeOutputShape(params.b, params.h, params.seqlen_q, q),
              /*num_outputs=*/3,
              torch::lazy::MHash(params_str)),
      params_(params),
      params_str_(params_str) {}

FlashAttentionForward::FlashAttentionForward(
    const torch::lazy::Value& q, const torch::lazy::Value& k,
    const torch::lazy::Value& v,
    const torch::lazy::Value& alibi_slopes,
    const FlashAttentionForwardParams& params,
    const std::string& params_str)
    : XlaNode(xla_flash_attention_forward,
              {q, k, v, alibi_slopes},
              NodeOutputShape(params.b, params.h, params.seqlen_q, q),
              /*num_outputs=*/3,
              torch::lazy::MHash(params_str)),
      params_(params),
      params_str_(params_str) {}

torch::lazy::NodePtr FlashAttentionForward::Clone(
    torch::lazy::OpList operands) const {
  if (operands.size() > 3){
    torch::lazy::MakeNode<FlashAttentionForward>(operands.at(0), operands.at(1),
                                                 operands.at(2), operands.at(3),
                                                 params_, params_str_);
  } else {
    torch::lazy::MakeNode<FlashAttentionForward>(operands.at(0), operands.at(1),
                                                 operands.at(2), params_, params_str_);
  }
}

XlaOpVector FlashAttentionForward::Lower(LoweringContext* loctx) const {
  xla::XlaOp q = loctx->GetOutputOp(operand(0));
  xla::XlaOp k = loctx->GetOutputOp(operand(1));
  xla::XlaOp v = loctx->GetOutputOp(operand(2));
  xla::XlaOp alibi_slopes =
      operands().size() > 3 ? loctx->GetOutputOp(operand(3)) : xla::XlaOp();
  std::vector<xla::XlaOp> result = BuildFlashAttentionForward(
      q, k, v, alibi_slopes, params_, xla_shape());
  return ReturnOps({result}, loctx);
}

}  // namespace torch_xla
