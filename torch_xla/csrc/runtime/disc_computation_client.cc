#include "torch_xla/csrc/runtime/disc_computation_client.h"

#include <vector>

#include "torch_xla/csrc/runtime/computation_client.h"

namespace torch_xla {
namespace runtime {

DISCComputationClient::DISCComputationClient() {}

DISCComputationClient::~DISCComputationClient() {}

void DISCComputationClient::DISCData::Assign(
    const torch::lazy::BackendData& data) {
  const DISCData& disc_data = dynamic_cast<const DISCData&>(data);
  if (&disc_data != this) {
    buffer = disc_data.buffer;
  }
}

ComputationClient::DataPtr DISCComputationClient::CreateDataPlaceholder(
    std::string device, xla::Shape shape,
    std::optional<xla::OpSharding> sharding) {
  return std::make_shared<DISCData>(std::move(device), std::move(shape));
}

std::vector<ComputationClient::DataPtr> TransferToDevice(
    absl::Span<const std::shared_ptr<const TensorSource>> tensors) {
  XLA_ERROR() << __FUNCTION__ << " not implemented";
}

std::vector<xla::Literal> DISCComputationClient::TransferFromDevice(
    absl::Span<const DataPtr> handles) {
  XLA_ERROR() << __FUNCTION__ << " not implemented";
}

std::vector<ComputationClient::ComputationPtr> DISCComputationClient::Compile(
    std::vector<ComputationClient::CompileInstance> instances) {
  XLA_ERROR() << __FUNCTION__ << " not implemented";
}

std::vector<ComputationClient::DataPtr>
DISCComputationClient::ExecuteComputation(
    const ComputationClient::Computation& computation,
    absl::Span<const ComputationClient::DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  XLA_ERROR() << __FUNCTION__ << " not implemented";
}

std::map<std::string, Metric> DISCComputationClient::GetMetrics() const {
  return {};
}

}  // namespace runtime
}  // namespace torch_xla
