

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
#include "torch_xla/csrc/flash_attention_utils.h"
#include "torch_xla/csrc/runtime/tf_logging.h"

namespace tao {
namespace ral {

DEFINE_TAO_TYPE_NAME_HELPER(Eigen::half, "f16");

template <typename T_IN, int M>
std::tuple<MemRefType<float, 3>, MemRefType<T_IN, M>, MemRefType<int64_t, 1>,
           MemRefType<int32_t, 1>, MemRefType<int32_t, 1>>
custom_call_flash_attention_varlen_forward_impl(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> q_memref,
    MemRefType<T_IN, M> k_memref, MemRefType<T_IN, M> v_memref,
    MemRefType<int32_t, 2> attention_mask_memref, void* alibi_slopes_ptr,
    void* customAttrs) {
  auto attr = getOrParsePDLAttr(ctx, customAttrs,
                                "custom_call_flash_attention_varlen_forward");
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
    output_element_count *= q_memref.sizes[i];
  }

  int bs = q_memref.sizes[0];
  int nheads = q_memref.sizes[2];
  int seqlen_q = q_memref.sizes[1];
  int seqlen_k = k_memref.sizes[1];

  auto softmax_lse_ptr = static_cast<float*>(
      gpu_driver->alloc(ctx, bs * nheads * seqlen_q * sizeof(float)));

  auto output_ptr = static_cast<T_IN*>(
      gpu_driver->alloc(ctx, output_element_count * sizeof(T_IN)));

  auto cu_seqlens_q_ptr =
      static_cast<int32_t*>(gpu_driver->alloc(ctx, (bs + 1) * sizeof(int32_t)));

  auto cu_seqlens_k_ptr =
      static_cast<int32_t*>(gpu_driver->alloc(ctx, (bs + 1) * sizeof(int32_t)));

  auto rng_state_ptr =
      static_cast<int64_t*>(gpu_driver->alloc(ctx, 2 * sizeof(int64_t)));

  torch_xla::FlashAttentionForwardParams params;
  params.FromString(std::move(backend_config));

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

  at::cuda::CUDAStreamGuard guard(
      at::cuda::getStreamFromExternal(gpu_stream, /*device_index=*/0));

  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

  at::Tensor q =
      torch::from_blob(q_memref.data, {bs * seqlen_q, params.h, params.d},
                       opts.dtype(scalar_type));
  at::Tensor k =
      torch::from_blob(k_memref.data, {bs * seqlen_k, params.h_k, params.d},
                       opts.dtype(scalar_type));
  at::Tensor v =
      torch::from_blob(v_memref.data, {bs * seqlen_k, params.h_k, params.d},
                       opts.dtype(scalar_type));
  at::Tensor attention_mask =
      torch::from_blob(attention_mask_memref.data, {bs, seqlen_k}, opts);
  at::Tensor softmax_lse = torch::from_blob(
      softmax_lse_ptr, {bs, params.h, seqlen_q}, opts.dtype(torch::kFloat));
  at::Tensor o_output =
      torch::from_blob(output_ptr, {bs * seqlen_q, params.h * params.d},
                       opts.dtype(scalar_type));
  at::Tensor cu_seqlens_q = torch::from_blob(cu_seqlens_q_ptr, {bs + 1}, opts);
  at::Tensor cu_seqlens_k = torch::from_blob(cu_seqlens_k_ptr, {bs + 1}, opts);
  at::Tensor rng_state =
      torch::from_blob(rng_state_ptr, {2}, opts.dtype(torch::kInt64));
  softmax_lse.fill_(0);
  o_output.fill_(0);
  cu_seqlens_k.fill_(0);

  int max_seqlen_in_batch_k = seqlen_k;
  int total_k = bs * seqlen_k;
  at::Tensor indices_k = torch_xla::mask_to_indices(
      attention_mask, max_seqlen_in_batch_k, total_k, cu_seqlens_k);

  auto unpad_k = torch_xla::index_first_axis(k, indices_k);
  auto unpad_v = torch_xla::index_first_axis(v, indices_k);

  int max_seqlen_in_batch_q = max_seqlen_in_batch_k;
  int total_q = total_k;
  at::Tensor indices_q;

  if (seqlen_q == seqlen_k) {
    cu_seqlens_q.copy_(cu_seqlens_k);
    indices_q = indices_k;
  } else if (seqlen_q == 1) {
    max_seqlen_in_batch_q = 1;
    cu_seqlens_q = torch::arange(0, bs + 1, opts);
    indices_q = cu_seqlens_q.slice(/*dim=*/0, /*start=*/0, /*end=*/bs);
    total_q = bs;
  } else {
    at::Tensor attention_mask_slice = attention_mask.slice(
        /*dim=*/1, /*start=*/-seqlen_q, /*end=*/torch::indexing::None);
    indices_q = torch_xla::mask_to_indices(
        attention_mask_slice, max_seqlen_in_batch_q, total_q, cu_seqlens_q);
  }
  at::Tensor unpad_q = torch_xla::index_first_axis(q, indices_q);

  at::Tensor unpad_output =
      torch::zeros({total_q, params.h * params.d}, opts.dtype(scalar_type));
  at::Tensor unpad_softmax_lse = torch::zeros(
      {bs, params.h, max_seqlen_in_batch_q}, opts.dtype(torch::kFloat));

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

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};

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

  launch_params.cu_seqlens_q = cu_seqlens_q_ptr;
  launch_params.cu_seqlens_k = cu_seqlens_k_ptr;

  launch_params.seqused_k = static_cast<int*>(nullptr);

  // P = softmax(QK^T)
  launch_params.p_ptr = nullptr;  // no softmax returned always

  // Softmax sum
  launch_params.softmax_lse_ptr = unpad_softmax_lse.data_ptr();

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

  launch_params.is_seqlens_k_cumulative = params.is_seqlens_k_cumulative;

  launch_params.alibi_slopes_ptr = alibi_slopes_ptr;
  launch_params.alibi_slopes_batch_stride = params.alibi_slopes_batch_stride;

  // set params splitkv
  launch_params.num_splits = params.num_splits;

  int64_t counter_offset = bs * params.h * 32;

  // Forward kernel will populate memory with the seed and offset.
  launch_params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

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
      run_mha_fwd_<elem_type, kHeadDim>(launch_params, gpu_stream);
    });
  });

  softmax_lse.slice(2, 0, max_seqlen_in_batch_q)
      .copy_(unpad_softmax_lse.slice(2, 0, max_seqlen_in_batch_q));

  torch::Tensor repeated_indices_q =
      indices_q.unsqueeze(1).expand({indices_q.size(0), params.h * params.d});
  o_output.scatter_(0, repeated_indices_q, unpad_output);

  auto softmax_lse_memref = assignMemRef<float, 3>(
      softmax_lse_ptr, std::vector<size_t>{bs, nheads, seqlen_q});
  auto output_memref = assignMemRef<T_IN, M>(output_ptr, q_memref.sizes);
  auto rng_state_memref =
      assignMemRef<int64_t, 1>(rng_state_ptr, std::vector<size_t>{2});
  auto cu_seqlens_q_memref =
      assignMemRef<int32_t, 1>(cu_seqlens_q_ptr, std::vector<size_t>{(bs + 1)});
  auto cu_seqlens_k_memref =
      assignMemRef<int32_t, 1>(cu_seqlens_k_ptr, std::vector<size_t>{(bs + 1)});

  return std::make_tuple(softmax_lse_memref, output_memref, rng_state_memref,
                         cu_seqlens_q_memref, cu_seqlens_k_memref);
}

template <typename T_IN, int M>
std::tuple<MemRefType<float, 3>, MemRefType<T_IN, M>, MemRefType<int64_t, 1>,
           MemRefType<int32_t, 1>, MemRefType<int32_t, 1>>
custom_call_flash_attention_varlen_forward_noalibi(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> q,
    MemRefType<T_IN, M> k, MemRefType<T_IN, M> v,
    MemRefType<int32_t, 2> attention_mask, void* customAttrs) {
  return custom_call_flash_attention_varlen_forward_impl<T_IN, M>(
      ctx, stream_handle, q, k, v, attention_mask, nullptr, customAttrs);
}

template <typename T_IN, int M>
std::tuple<MemRefType<float, 3>, MemRefType<T_IN, M>, MemRefType<int64_t, 1>,
           MemRefType<int32_t, 1>, MemRefType<int32_t, 1>>
custom_call_flash_attention_varlen_forward_alibi_v1(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> q,
    MemRefType<T_IN, M> k, MemRefType<T_IN, M> v,
    MemRefType<int32_t, 2> attention_mask, MemRefType<float, 1> alibi_slopes,
    void* customAttrs) {
  return custom_call_flash_attention_varlen_forward_impl<T_IN, M>(
      ctx, stream_handle, q, k, v, attention_mask, alibi_slopes.data,
      customAttrs);
}

template <typename T_IN, int M>
std::tuple<MemRefType<float, 3>, MemRefType<T_IN, M>, MemRefType<int64_t, 1>,
           MemRefType<int32_t, 1>, MemRefType<int32_t, 1>>
custom_call_flash_attention_varlen_forward_alibi_v2(
    ExecutionContext* ctx, void* stream_handle, MemRefType<T_IN, M> q,
    MemRefType<T_IN, M> k, MemRefType<T_IN, M> v,
    MemRefType<int32_t, 2> attention_mask, MemRefType<float, 2> alibi_slopes,
    void* customAttrs) {
  return custom_call_flash_attention_varlen_forward_impl<T_IN, M>(
      ctx, stream_handle, q, k, v, attention_mask, alibi_slopes.data,
      customAttrs);
}

TAO_RAL_API("custom_call_flash_attention_varlen_forward", "gpu",
            custom_call_flash_attention_varlen_forward_noalibi<Eigen::half, 4>);
TAO_RAL_API(
    "custom_call_flash_attention_varlen_forward", "gpu",
    custom_call_flash_attention_varlen_forward_alibi_v1<Eigen::half, 4>);
TAO_RAL_API(
    "custom_call_flash_attention_varlen_forward", "gpu",
    custom_call_flash_attention_varlen_forward_alibi_v2<Eigen::half, 4>);
TAO_RAL_API("custom_call_flash_attention_varlen_forward", "gpu",
            custom_call_flash_attention_varlen_forward_noalibi<bfloat16, 4>);
TAO_RAL_API("custom_call_flash_attention_varlen_forward", "gpu",
            custom_call_flash_attention_varlen_forward_alibi_v1<bfloat16, 4>);
TAO_RAL_API("custom_call_flash_attention_varlen_forward", "gpu",
            custom_call_flash_attention_varlen_forward_alibi_v2<bfloat16, 4>);

}  // namespace ral
}  // namespace tao
