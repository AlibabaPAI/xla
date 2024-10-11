#ifndef XLA_TORCH_XLA_CSRC_RUNTIME_DISC_DISCRAL_H_
#define XLA_TORCH_XLA_CSRC_RUNTIME_DISC_DISCRAL_H_

#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <ral/context/base/cuda/cuda_context_impl.h>

#include "torch_xla/csrc/runtime/disc/disc_utils.h"

namespace torch_xla {
namespace runtime {
namespace disc {

using tao::ral::ExecutionContext;

struct DataMeta {
  std::string device;
  c10::ScalarType scalar_type;
};

struct DISCComplationResult {
  std::string ral_lib;
  std::string ral_mate_pb;
  std::vector<DataMeta> inputs;
  std::vector<DataMeta> outputs;
};

class RalContext {
  using EntryFunc = std::function<void(void**)>;

 public:
  RalContext(const DISCComplationResult& disc_result);
  ~RalContext();

  std::vector<at::Tensor> Execute(const std::vector<at::Tensor>& inputs);

 private:
  void BindingInputs(const std::vector<at::Tensor>& inputs,
                     tao::ral::ExecutionContext& exec_ctx);
  void CheckCurrentDevice(const std::vector<at::Tensor>& inputs);
  std::vector<at::Tensor> CreateAndBindingOutputs(
      const std::vector<at::Tensor>& inputs,
      tao::ral::ExecutionContext& exec_ctx);
  std::vector<at::Tensor> PreProcessInputs(
      const std::vector<at::Tensor>& inputs);
  std::tuple<void*, void*> LoadEngine(const std::string& ral_engine_bytes);

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

  DISCComplationResult disc_result_;

  void* tao_lib_;
  EntryFunc entry_func_;

  TempFile lib_tmpf_{"ral_lib.so"};
  TempFile meta_tmpf_{"ral_meta.pb"};
};
}  // namespace disc
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_RUNTIME_DISC_DISCRAL_H_
