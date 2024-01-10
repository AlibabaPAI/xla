#ifndef XLA_TORCH_XLA_CSRC_RUNTIME_DISC_COMPILE_H_
#define XLA_TORCH_XLA_CSRC_RUNTIME_DISC_COMPILE_H_

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "torch_xla/csrc/runtime/disc/disc_ral.h"

namespace torch_xla {
namespace runtime {
namespace disc {
DISCComplationResult Compile(mlir::ModuleOp& module,
                             const std::vector<DataMeta>& inputs,
                             const std::vector<DataMeta>& outputs);

}  // namespace disc
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_RUNTIME_DISC_COMPILE_H_