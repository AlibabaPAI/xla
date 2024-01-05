#include <ATen/cuda/CUDAContext.h>
#include <ral/context/base/base_context.h>
#include <ral/context/base/cuda/cuda_context_impl.h>
#include <torch/script.h>

namespace torch_xla {
namespace runtime {

using tao::ral::BaseContext;
using tao::ral::ExecutionContext;

class DataMeta {
 public:
  std::string device;
  c10::ScalarType scalar_type;
};

class DISCComplationResult {
 public:
  std::string ral_lib;
  std::string ral_mate_pb;
  std::vector<DataMeta> inputs;
  std::vector<DataMeta> outputs;
};

class RalContext {
  using EntryFunc = std::function<void(void**)>;

 public:
  RalContext(std::shared_ptr<DISCComplationResult> disc_result)
      : disc_result_(disc_result){};
  ~RalContext(){};

  at::List<at::Tensor> Execute(const at::List<at::Tensor>&);

 private:
  void BindingInputs(const at::List<at::Tensor>& inputs,
                     tao::ral::ExecutionContext& exec_ctx);
  void CheckCurrentDevice(const at::List<at::Tensor>& inputs);
  at::List<at::Tensor> CreateAndBindingOutputs(
      tao::ral::ExecutionContext& exec_ctx);
  at::List<at::Tensor> PreProcessInputs(const at::List<at::Tensor>& inputs);

  int64_t LazyInitCurrentDevice();

  constexpr static int64_t NULL_GPU_DEVICE = -1;
  std::atomic<int64_t> gpu_device_{NULL_GPU_DEVICE};
  std::mutex mtx_;
  std::unordered_map<c10::cuda::CUDAStream,
                     std::unique_ptr<tao::ral::BaseContext>>
      ral_ctx_map_;
  tao::ral::BaseContext* LoadCache();

  tao::ral::BaseContextOption default_opt_;
  tao::ral::cpu::BaseCpuContextOption cpu_opt_;

  std::shared_ptr<DISCComplationResult> disc_result_;

  void* tao_lib_;
  EntryFunc entry_func_;
};
}  // namespace runtime
}  // namespace torch_xla
