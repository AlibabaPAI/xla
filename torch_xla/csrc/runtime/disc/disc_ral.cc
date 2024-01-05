#include "torch_xla/csrc/runtime/disc/disc_ral.h"

#include <c10/core/impl/alloc_cpu.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <ral/context/base/cuda/cuda_context_impl.h>
#include <ral/ral_api.h>
#include <torch/torch.h>
namespace torch_xla {
namespace runtime {
class RalAllocator : public tao::ral::Allocator {
 public:
  using buffer_t = tao::ral::buffer_t;
  using alloc_t = tao::ral::alloc_t;
  using dealloc_t = tao::ral::dealloc_t;
  RalAllocator(alloc_t alloc_func, dealloc_t dealloc_func)
      : alloc_func_(alloc_func), dealloc_func_(dealloc_func) {}

  buffer_t alloc(size_t bytes) { return alloc_func_(bytes); }

  void dealloc(buffer_t buffer) { dealloc_func_(buffer); }

 private:
  alloc_t alloc_func_;
  dealloc_t dealloc_func_;
};

int64_t RalContext::LazyInitCurrentDevice() {
  int64_t cur_device = c10::cuda::current_device();
  int64_t prev_device = NULL_GPU_DEVICE;
  bool success = gpu_device_.compare_exchange_strong(prev_device, cur_device);
  if (!success) {
    TORCH_CHECK(prev_device == cur_device,
                "Device changed during inference. Please do NOT change CUDA "
                "current device during inference.");
  }
  TORCH_CHECK(gpu_device_ != NULL_GPU_DEVICE);
  return cur_device;
}

BaseContext* RalContext::LoadCache() {
  int64_t gpu_device = LazyInitCurrentDevice();
  TORCH_CHECK(gpu_device >= 0, "expect gpu device id >= 0, but got ",
              gpu_device);
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(gpu_device);

  // TODO: take care of the duplicated const
  // which currently is managed per context
  tao::ral::gpu::BaseCudaContextOption gpu_opt;
  gpu_opt.device_ordinal = gpu_device;
  gpu_opt.use_stream_executor = true;
  gpu_opt.gpu_allocator.reset(
      new RalAllocator(c10::cuda::CUDACachingAllocator::raw_alloc,
                       c10::cuda::CUDACachingAllocator::raw_delete));

  std::lock_guard<std::mutex> guard(mtx_);
  tao::ral::BaseContext* ral_ctx_ptr;
  auto it = ral_ctx_map_.find(stream);
  if (it == ral_ctx_map_.end()) {
    gpu_opt.stream = stream.stream();
    auto ral_ctx =
        tao::ral::gpu::MakeBaseCudaContext(default_opt_, cpu_opt_, gpu_opt);
    ral_ctx_ptr = ral_ctx.get();
    ral_ctx_map_[stream].reset(ral_ctx.release());
  } else {
    ral_ctx_ptr = it->second.get();
  }
  return ral_ctx_ptr;
}
at::List<at::Tensor> RalContext::PreProcessInputs(
    const at::List<at::Tensor>& inputs) {
  at::List<at::Tensor> contiguous_inputs;
  for (at::Tensor inp_tensor : inputs) {
    // make sure the input is in contiguous layout
    contiguous_inputs.push_back(inp_tensor.contiguous());
  }
  return contiguous_inputs;
}

inline bool IsEmptyTensor(const tao::ral::buffer_shape_t& shape) {
  return shape.size() > 0 && std::any_of(shape.begin(), shape.end(),
                                         [](int64_t dim) { return dim == 0; });
}

at::List<at::Tensor> RalContext::CreateAndBindingOutputs(
    tao::ral::ExecutionContext& exec_ctx) {
  at::List<at::Tensor> outputs;

  auto num_outputs = disc_result_->outputs.size();
  outputs.reserve(num_outputs);
  std::vector<std::unique_ptr<tao::ral::OutputBufferWrapper>> out_bufs(
      num_outputs);
  for (size_t idx = 0; idx < num_outputs; ++idx) {
    auto& out_buf = out_bufs[idx];
    // Note: Ral has memory allocator that allocate memory each time forward.
    // So it's thread-safe to reuse the underline memory.
    exec_ctx.bindOutput(idx, &out_buf);

    const auto& output_info = disc_result_->outputs[idx];
    auto scalar_type = output_info.scalar_type;
    torch::DeviceType dev_type = torch::kCUDA;
    dev_type = (output_info.device == "cuda") ? torch::kCUDA : torch::kCPU;

    auto option = torch::device(dev_type)
                      .dtype(scalar_type)
                      .memory_format(torch::MemoryFormat::Contiguous);
    at::Tensor out_tensor;
    if (IsEmptyTensor(out_buf->shape())) {
      out_tensor = torch::zeros(out_buf->shape(), option);
    } else if (out_buf->owned()) {
      auto cpu_allocator = c10::GetAllocator(torch::kCPU);
      TORCH_CHECK(cpu_allocator != nullptr);
      std::function<void(void*)> deleter = [cpu_allocator](void* ptr) {
        cpu_allocator->raw_deallocate(ptr);
      };
      if (output_info.device == "cuda") {
        deleter = c10::cuda::CUDACachingAllocator::raw_delete;
      }
      out_tensor = torch::from_blob(const_cast<void*>(out_buf->data()),
                                    out_buf->shape(), deleter, option);
      out_buf->release();
    } else {
      out_tensor = torch::from_blob(const_cast<void*>(out_buf->data()),
                                    out_buf->shape(), option)
                       .clone();
    }
    outputs.push_back(out_tensor);
  }
  return outputs;
}
void RalContext::BindingInputs(const at::List<at::Tensor>& inputs,
                               tao::ral::ExecutionContext& exec_ctx) {
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    at::Tensor inp = inputs[idx];
    const auto& shape = inp.sizes();
    exec_ctx.bindInput(idx, inp.data_ptr(), shape.vec());
  }
}
at::List<at::Tensor> RalContext::Execute(const at::List<at::Tensor>& inputs) {
  auto ral_ctx = LoadCache();
  // execution context is per-inference context and thread-safe
  auto exec_ctx =
      tao::ral::MakeExecutionContext<tao::ral::gpu::BaseCudaExecutionContext>(
          ral_ctx);

  auto contiguous_inputs = PreProcessInputs(inputs);
  BindingInputs(contiguous_inputs, *exec_ctx.get());

  auto tao_ral_func_ptr = reinterpret_cast<void*>(&tao_ral_call_impl);

  // execute
  void* ctx_struct[] = {exec_ctx.get(), tao_ral_func_ptr};
  try {
    entry_func_(ctx_struct);
  } catch (std::exception& ex) {
    LOG(ERROR) << ex.what();
    throw ex;
  }

  auto outputs = CreateAndBindingOutputs(*exec_ctx.get());
  return outputs;
}

}  //  namespace runtime
}  //  namespace torch_xla
