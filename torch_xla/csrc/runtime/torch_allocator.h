#ifndef XLA_CLIENT_TORCH_ALLOCATOR_H_
#define XLA_CLIENT_TORCH_ALLOCATOR_H_

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include "tsl/framework/allocator.h"

namespace torch_xla {
namespace runtime {

class TorchCUDACachingAllocator : public tsl::Allocator {
 public:
  TorchCUDACachingAllocator(int device_ordinal);
  ~TorchCUDACachingAllocator() override{};

  std::string Name() override { return name_; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;

  void SetStreamAndPreallocateMemory(void* stream) override;

  tsl::AllocatorMemoryType GetMemoryType() const override {
    return tsl::AllocatorMemoryType::kDevice;
  }

 private:
  std::string name_;
  cudaStream_t cuda_stream_;
  c10::DeviceIndex device_index_;
};

}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_TORCH_ALLOCATOR_H_
