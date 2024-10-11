#include "torch_xla/csrc/runtime/disc/disc_ral.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <dlfcn.h>
#include <ral/context/base/cuda/cuda_context_impl.h>
#include <ral/ral_api.h>
#include <ral/ral_context.h>
#include <torch/torch.h>

#include <chrono>

#include "absl/strings/str_cat.h"
#include "torch_xla/csrc/runtime/tf_logging.h"

namespace torch_xla {
namespace runtime {
namespace disc {

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

RalContext::RalContext(const DISCComplationResult& disc_result)
    : disc_result_(disc_result) {
  auto is_ok = meta_tmpf_.WriteBytesToFile(disc_result_.ral_mate_pb);
  TORCH_CHECK(is_ok, "Failed to dump model proto to file.");
  default_opt_.metadata_file_path = meta_tmpf_.GetFilename();
  default_opt_.cache_workspace_mem_across_execution = true;
  auto torch_allocator = c10::GetAllocator(torch::kCPU);
  TORCH_CHECK(torch_allocator != nullptr);
  auto cpu_alloc = [torch_allocator](size_t n) {
    return torch_allocator->raw_allocate(n);
  };
  auto cpu_delete = [torch_allocator](void* ptr) {
    torch_allocator->raw_deallocate(ptr);
  };
  cpu_opt_.cpu_allocator.reset(new RalAllocator(cpu_alloc, cpu_delete));

  at::globalContext().lazyInitCUDA();

  void* func_handle = nullptr;
  std::tie(tao_lib_, func_handle) = LoadEngine(disc_result_.ral_lib);

  using func_t = void (*)(void**);
  entry_func_ = (func_t)func_handle;

  CHECK(entry_func_ != nullptr);
}

std::tuple<void*, void*> RalContext::LoadEngine(
    const std::string& ral_engine_bytes) {
  auto is_ok = lib_tmpf_.WriteBytesToFile(ral_engine_bytes);
  TORCH_CHECK(is_ok, "Failed to dump RAL engine to file");
  std::string filename = lib_tmpf_.GetFilename();

  void* tao_lib = dlopen(filename.c_str(), RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(tao_lib, "Fail to open ral engine");

  void* func_handle = dlsym(tao_lib, kMlirLoweredEntry);
  TORCH_CHECK(func_handle, "Fail to find kMlirLoweredEntry");
  return std::make_tuple(tao_lib, func_handle);
}

RalContext::~RalContext() {
  if (tao_lib_ != nullptr) {
    dlclose(tao_lib_);
  }
}

void RalContext::CheckCurrentDevice(const std::vector<at::Tensor>& inputs) {
  int64_t gpu_device = LazyInitCurrentDevice();
  // Engine Context
  if (inputs.empty()) {
    return;
  }

  torch::Device cur_cuda_device = torch::Device(torch::kCUDA, gpu_device);

  TORCH_CHECK(disc_result_.inputs.size() == inputs.size());
  for (size_t k = 0; k < inputs.size(); ++k) {
    at::Tensor inp = inputs[k];
    auto device = disc_result_.inputs[k].device;
    if (device == "cuda") {
      TORCH_CHECK(inp.device() == cur_cuda_device, "Input tensor ", k,
                  " device mismatch. Expect: ", cur_cuda_device,
                  ", got: ", inp.device());
    }
  }
  return;
}

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

tao::ral::BaseContext* RalContext::LoadCache() {
  int64_t gpu_device = LazyInitCurrentDevice();
  TORCH_CHECK(gpu_device >= 0, "expect gpu device id >= 0, but got ",
              gpu_device);
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(gpu_device);

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

std::vector<at::Tensor> RalContext::PreProcessInputs(
    const std::vector<at::Tensor>& inputs) {
  CheckCurrentDevice(inputs);

  std::vector<at::Tensor> contiguous_inputs;
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

inline bool IsSameShape(const tao::ral::buffer_shape_t& shape,
                        at::Tensor input_tensor) {
  if (input_tensor.dim() != shape.size()) {
    return false;
  }

  for (int i = 0; i < shape.size(); i++) {
    if (input_tensor.sizes()[i] != shape[i]) {
      return false;
    }
  }

  return true;
}

std::vector<at::Tensor> RalContext::CreateAndBindingOutputs(
    const std::vector<at::Tensor>& inputs,
    tao::ral::ExecutionContext& exec_ctx) {
  std::vector<at::Tensor> outputs;

  auto num_outputs = disc_result_.outputs.size();
  outputs.reserve(num_outputs);
  std::vector<std::unique_ptr<tao::ral::OutputBufferWrapper>> out_bufs(
      num_outputs);
  for (size_t idx = 0; idx < num_outputs; ++idx) {
    auto& out_buf = out_bufs[idx];
    // Note: Ral has memory allocator that allocate memory each time forward.
    // So it's thread-safe to reuse the underline memory.
    exec_ctx.bindOutput(idx, &out_buf);

    const auto& output_info = disc_result_.outputs[idx];
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
      //(@yuanxiulong.yxl) For input output alias, now we will only have full
      // tensor memory reuse.
      // We will support partial memory space reuse in the future
      bool alias_input = false;
      for (auto& input_tensor : inputs) {
        // same address, same shape, same dtype
        if (input_tensor.data_ptr() == out_buf->data() &&
            scalar_type == input_tensor.dtype() &&
            IsSameShape(out_buf->shape(), input_tensor)) {
          out_tensor = input_tensor;
          alias_input = true;
        }
      }
      if (!alias_input) {
        out_tensor = torch::from_blob(const_cast<void*>(out_buf->data()),
                                      out_buf->shape(), option)
                         .clone();
      }
    }
    outputs.push_back(out_tensor);
  }
  return outputs;
}

void RalContext::BindingInputs(const std::vector<at::Tensor>& inputs,
                               tao::ral::ExecutionContext& exec_ctx) {
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    at::Tensor inp = inputs[idx];
    const auto& shape = inp.sizes();
    exec_ctx.bindInput(idx, inp.data_ptr(), shape.vec());
  }
}

std::vector<at::Tensor> RalContext::Execute(
    const std::vector<at::Tensor>& inputs) {
  // inputs are always contigous
  auto ral_ctx = LoadCache();
  // execution context is per-inference context and thread-safe
  auto exec_ctx =
      tao::ral::MakeExecutionContext<tao::ral::gpu::BaseCudaExecutionContext>(
          ral_ctx);

  BindingInputs(inputs, *exec_ctx.get());

  auto tao_ral_func_ptr = reinterpret_cast<void*>(&tao_ral_call_impl);

  // execute
  void* ctx_struct[] = {exec_ctx.get(), tao_ral_func_ptr};
  try {
    entry_func_(ctx_struct);
  } catch (std::exception& ex) {
    LOG(ERROR) << ex.what();
    throw ex;
  }

  // Support input output buffer reuse
  // Now we only have full buffer reuse for alias
  auto outputs = CreateAndBindingOutputs(inputs, *exec_ctx.get());

  return outputs;
}

}  // namespace disc
}  // namespace runtime
}  // namespace torch_xla
