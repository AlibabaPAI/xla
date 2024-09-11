#include "torch_xla/csrc/runtime/torch_allocator.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>

#include "torch_xla/csrc/runtime/tf_logging.h"

namespace torch_xla {
namespace runtime {

TorchCUDACachingAllocator::TorchCUDACachingAllocator(int device_ordinal) {
  VLOG(3) << "Creating TorchCUDACachingAllocator for device " << device_ordinal;
  name_ = c10::cuda::CUDACachingAllocator::name();
  cuda_stream_ = nullptr;
  device_index_ = static_cast<c10::DeviceIndex>(device_ordinal);
}

void* TorchCUDACachingAllocator::AllocateRaw(size_t alignment,
                                             size_t num_bytes) {
  CHECK(cuda_stream_ != nullptr)
      << "A stream must be added to the TorchCUDACachingAllocator allocator";
  if (num_bytes == 0) {
    return nullptr;
  }
  at::cuda::CUDAGuard device_guard{device_index_};
  auto ptr = c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(
      num_bytes, cuda_stream_);
  VLOG(3) << "Alloc num_bytes " << num_bytes << " with ptr " << ptr
          << " for device " << static_cast<int>(device_index_);
  return ptr;
}

void TorchCUDACachingAllocator::DeallocateRaw(void* ptr) {
  VLOG(3) << "Dealloc ptr " << ptr << " for device "
          << static_cast<int>(device_index_);
  c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}

void TorchCUDACachingAllocator::SetStreamAndPreallocateMemory(void* stream) {
  auto new_cuda_stream = static_cast<cudaStream_t>(stream);
  if (cuda_stream_ != nullptr && new_cuda_stream != cuda_stream_) {
    LOG(FATAL) <<  // Crash OK.
        "Trying to set the stream twice. This isn't supported. ";
  }
  VLOG(3) << "Setting cuda stream " << stream
          << " for TorchCUDACachingAllocator on device "
          << static_cast<int>(device_index_);
  cuda_stream_ = new_cuda_stream;
}

}  // namespace runtime
}  // namespace torch_xla
