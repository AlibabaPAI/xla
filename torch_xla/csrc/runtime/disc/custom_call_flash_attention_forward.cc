

#include <c10/cuda/CUDAStream.h>
#include <ral/context/base/cuda/cuda_context_impl.h>
#include <ral/context/context_util.h>
#include <ral/device/gpu/gpu_driver.h>
#include <ral/ral_api.h>
#include <ral/ral_context.h>
#include <ral/ral_driver.h>
#include <ral/ral_helper.h>
#include <ral/ral_logging.h>

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

struct FlashAttentionForwardParams {
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

  void FromString(const std::string& str) {
    std::vector<std::string> params_list = absl::StrSplit(str, "|");
    TORCH_CHECK(params_list.size() >= 38);  // at least 38 variables
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
  }
};

// Layout of `buffers` listed above:
//  buffers[0] = q
//  buffers[1] = k
//  buffers[2] = v
//  buffers[3] = cu_seqlens_q
//  buffers[4] = cu_seqlens_k
//  result[0] = softmax_lse  // this is output
//  result[1] = out_for_output // this is output
template <typename T_IN, typename SOFT_MAX_TYPE, int M>
std::tuple<MemRefType<SOFT_MAX_TYPE, M>, MemRefType<T_IN, M>,
           MemRefType<int64_t, 1>>
custom_call_flash_attention_forward_impl(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> q,
    MemRefType<T_IN, M> k, MemRefType<T_IN, M> v,
    MemRefType<int32_t, 1> seqlens_q, MemRefType<int32_t, 1> seqlens_k,
    void* alibi_slopes_ptr, void* customAttrs) {
  auto attr = getOrParsePDLAttr(ctx, customAttrs,
                                "custom_call_flash_attention_forward");
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

  int output_element_count = 1;
  for (int i = 0; i < M; i++) {
    output_element_count *= q.sizes[i];
  }

  int bs = seqlens_q.sizes[0] - 1;
  int nheads = q.sizes[1];
  int seqlen = q.sizes[0] / bs;
  std::vector<size_t> softmax_lse_sizes{bs, nheads, seqlen};

  auto softmax_lse_ptr = static_cast<SOFT_MAX_TYPE*>(
      gpu_driver->alloc(ctx, bs * nheads * seqlen * sizeof(SOFT_MAX_TYPE)));
  auto softmax_lse =
      assignMemRef<SOFT_MAX_TYPE, M>(softmax_lse_ptr, softmax_lse_sizes);

  auto output_ptr = static_cast<T_IN*>(
      gpu_driver->alloc(ctx, output_element_count * sizeof(T_IN)));
  auto output = assignMemRef<T_IN, M>(output_ptr, q.sizes);

  auto rng_state_ptr =
      static_cast<int64_t*>(gpu_driver->alloc(ctx, 2 * sizeof(int64_t)));
  auto rng_state =
      assignMemRef<int64_t, 1>(rng_state_ptr, std::vector<size_t>{2});

  cudaMemsetAsync(rng_state_ptr, 0, 2 * sizeof(int64_t), gpu_stream);

  FlashAttentionForwardParams params;
  params.FromString(std::move(backend_config));

  Flash_fwd_params launch_params;

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
  launch_params.o_ptr = output.data;
  launch_params.o_row_stride = params.o_row_stride;
  launch_params.o_head_stride = params.o_head_stride;

  launch_params.cu_seqlens_q = seqlens_q.data;
  launch_params.cu_seqlens_k = seqlens_k.data;
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

  launch_params.is_seqlens_k_cumulative = params.is_seqlens_k_cumulative;

  // set params splitkv
  launch_params.num_splits = params.num_splits;

  // Forward kernel will populate memory with the seed and offset.
  launch_params.rng_state = reinterpret_cast<uint64_t*>(rng_state_ptr);

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

  FP16_SWITCH(!launch_params.is_bf16, [&] {
    HEADDIM_SWITCH(launch_params.d, [&] {
      // TODO(wenting.swt): support split_kv
      run_mha_fwd_<elem_type, kHeadDim>(launch_params, gpu_stream);
    });
  });

  return std::make_tuple(softmax_lse, output, rng_state);
}

template <typename T_IN, typename SOFT_MAX_TYPE, int M>
std::tuple<MemRefType<SOFT_MAX_TYPE, M>, MemRefType<T_IN, M>,
           MemRefType<int64_t, 1>>
custom_call_flash_attention_forward_noalibi(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> q,
    MemRefType<T_IN, M> k, MemRefType<T_IN, M> v,
    MemRefType<int32_t, 1> seqlens_q, MemRefType<int32_t, 1> seqlens_k,
    void* customAttrs) {
  return custom_call_flash_attention_forward_impl<T_IN, SOFT_MAX_TYPE, M>(
      ctx, stream_handle, q, k, v, seqlens_q, seqlens_k, nullptr, customAttrs);
}

template <typename T_IN, typename SOFT_MAX_TYPE, int M>
std::tuple<MemRefType<SOFT_MAX_TYPE, M>, MemRefType<T_IN, M>,
           MemRefType<int64_t, 1>>
custom_call_flash_attention_forward_alibi_v1(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> q,
    MemRefType<T_IN, M> k, MemRefType<T_IN, M> v,
    MemRefType<int32_t, 1> seqlens_q, MemRefType<int32_t, 1> seqlens_k,
    MemRefType<float, 1> alibi_slopes, void* customAttrs) {
  return custom_call_flash_attention_forward_impl<T_IN, SOFT_MAX_TYPE, M>(
      ctx, stream_handle, q, k, v, seqlens_q, seqlens_k, alibi_slopes.data,
      customAttrs);
}

template <typename T_IN, typename SOFT_MAX_TYPE, int M>
std::tuple<MemRefType<SOFT_MAX_TYPE, M>, MemRefType<T_IN, M>,
           MemRefType<int64_t, 1>>
custom_call_flash_attention_forward_alibi_v2(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> q,
    MemRefType<T_IN, M> k, MemRefType<T_IN, M> v,
    MemRefType<int32_t, 1> seqlens_q, MemRefType<int32_t, 1> seqlens_k,
    MemRefType<float, 2> alibi_slopes, void* customAttrs) {
  return custom_call_flash_attention_forward_impl<T_IN, SOFT_MAX_TYPE, M>(
      ctx, stream_handle, q, k, v, seqlens_q, seqlens_k, alibi_slopes.data,
      customAttrs);
}

TAO_RAL_API("custom_call_flash_attention_forward", "gpu",
            custom_call_flash_attention_forward_noalibi<Eigen::half, float, 3>);
TAO_RAL_API(
    "custom_call_flash_attention_forward", "gpu",
    custom_call_flash_attention_forward_alibi_v1<Eigen::half, float, 3>);
TAO_RAL_API(
    "custom_call_flash_attention_forward", "gpu",
    custom_call_flash_attention_forward_alibi_v2<Eigen::half, float, 3>);
TAO_RAL_API("custom_call_flash_attention_forward", "gpu",
            custom_call_flash_attention_forward_noalibi<bfloat16, float, 3>);
TAO_RAL_API("custom_call_flash_attention_forward", "gpu",
            custom_call_flash_attention_forward_alibi_v1<bfloat16, float, 3>);
TAO_RAL_API("custom_call_flash_attention_forward", "gpu",
            custom_call_flash_attention_forward_alibi_v2<bfloat16, float, 3>);

}  // namespace ral
}  // namespace tao
