#include "torch_xla/csrc/runtime/disc/disc_ral.h"

#include <gtest/gtest.h>

namespace torch_xla {
namespace runtime {
namespace disc {
TEST(DISCRAlTest, E2E) {
  // TODO(disc): need compile API to output the compilation result
  std::shared_ptr<DISCComplationResult> disc_result =
      std::make_shared<DISCComplationResult>();
  RalContext ral_ctx(disc_result);
  std::vector<at::Tensor> inputs;
  ral_ctx.Execute(at::List<at::Tensor>());
}

}  // namespace disc
}  // namespace runtime
}  // namespace torch_xla
