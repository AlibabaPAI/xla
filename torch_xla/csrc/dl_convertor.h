#ifndef XLA_TORCH_XLA_CSRC_DL_CONVERTOR_H_
#define XLA_TORCH_XLA_CSRC_DL_CONVERTOR_H_

#include <ATen/Tensor.h>
#include <ATen/dlpack.h>

namespace torch_xla {

DLManagedTensor* toDLPack(const at::Tensor& src);
DLManagedTensor* toDLPackAlias(const at::Tensor& input,
                               const at::Tensor& result, int64_t address);
at::Tensor fromDLPack(DLManagedTensor* src);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_DL_CONVERTOR_H_