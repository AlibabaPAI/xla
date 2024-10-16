#ifndef XLA_CLIENT_DISC_COMPUTATION_CLIENT_H_
#define XLA_CLIENT_DISC_COMPUTATION_CLIENT_H_

#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/disc/disc_ral.h"
#include "torch_xla/csrc/runtime/stablehlo_helper.h"
#include "xla/client/xla_computation.h"

namespace torch_xla {
namespace runtime {

class DISCComputationClient : public ComputationClient {
 public:
  DISCComputationClient();
  ~DISCComputationClient();

  DataPtr CreateDataPlaceholder(
      std::string device, xla::Shape shape,
      std::optional<xla::OpSharding> sharding = std::nullopt) override;

  std::vector<DataPtr> GetDataShards(DataPtr data) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  DataPtr GetDataShard(DataPtr data, size_t index) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  std::vector<DataPtr> ReshardData(
      absl::Span<const DataPtr> handles,
      absl::Span<const xla::OpSharding> shardings) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  DataPtr WrapDataShards(absl::Span<const DataPtr> shards, std::string device,
                         xla::Shape shape, xla::OpSharding sharding) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  std::optional<xla::OpSharding> GetDataSharding(DataPtr handle) override;

  std::vector<DataPtr> TransferToDevice(
      absl::Span<const std::shared_ptr<const TensorSource>> tensors) override;

  std::vector<xla::Literal> TransferFromDevice(
      absl::Span<const DataPtr> handles) override;

  DataPtr TransferShardsToDevice(
      absl::Span<const std::shared_ptr<const TensorSource>> tensor_shards,
      std::string device, xla::Shape shape, xla::OpSharding sharding) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  DataPtr CopyToDevice(DataPtr data, std::string dst) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  std::string SerializeComputation(const ComputationPtr computation) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  ComputationPtr DeserializeComputation(
      const std::string& serialized) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  torch::lazy::hash_t HashCompilationEnv() override {
    // TODO(wangang.wa): Improve this function.
    return torch::lazy::hash_t();
  }

  torch_xla::DeviceType GetDeviceType() const override {
    return torch_xla::DeviceType("CUDA");
  };

  bool CoordinatorInitialized() const override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  void InitializeCoordinator(int global_rank, int world_size,
                             std::string master_addr,
                             std::string port) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  XlaCoordinator& GetCoordinator() override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const std::string& device,
      const ExecuteComputationOptions& options) override;

  std::vector<DataPtr> ExecuteReplicated(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      absl::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  size_t GetNumDevices() const override;

  std::string GetDefaultDevice() const override;

  std::vector<std::string> GetLocalDevices() const override;

  std::vector<std::string> GetAllDevices() const override;

  int GetProcessIndex() const override;

  int GetNumProcesses() const override;

  const absl::flat_hash_map<
      std::string, torch_xla::runtime::ComputationClient::DeviceAttribute>&
  GetDeviceAttributes(const std::string& device) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) override;

  std::shared_ptr<std::vector<std::string>> GetReplicationDevices() override;

  void WaitDeviceOps(absl::Span<const std::string> devices) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

  std::map<std::string, Metric> GetMetrics() const override;

  MemoryInfo GetMemoryInfo(const std::string& device) override {
    XLA_ERROR() << __FUNCTION__ << " not implemented";
  }

 private:
  std::shared_ptr<std::vector<std::string>> replication_devices_;
  int world_size_;
  int local_rank_;
  int global_rank_;
  std::string device_type_;
  struct DISCData : public Data {
    DISCData(std::string device, xla::Shape device_shape)
        : Data(std::move(device), std::move(device_shape)) {}

    DISCData(std::string device, xla::Shape device_shape, at::Tensor buffer)
        : Data(std::move(device), std::move(device_shape)), buffer(buffer) {}

    void Assign(const torch::lazy::BackendData& data) override;

    bool HasValue() const override {
      return buffer.defined() && buffer.element_size() > 0;
    }

    Handle GetHandle() override {
      return reinterpret_cast<std::uintptr_t>(buffer.const_data_ptr());
    }

    bool HasSharding() const override { return false; }

    xla::OpSharding GetSharding() const override {
      XLA_CHECK(false) << "GetSharding should not be called on DISCData, check "
                          "HasSharding first";
      return xla::OpSharding();
    }

    std::string ToString() const override {
      std::stringstream ss;
      ss << "XLAData: \n";
      ss << "  Data Device: " << device() << "\n";
      ss << "  Data Shape: " << shape().ToString() << "\n";
      ss << "  Data Handle: ";
      if (HasValue()) {
        ss << reinterpret_cast<std::uintptr_t>(buffer.const_data_ptr()) << "\n";
      } else {
        ss << "None\n";
      }
      return ss.str();
    }

    at::Tensor buffer;
  };

  struct DISCComputation : public Computation {
    DISCComputation(xla::XlaComputation computation,
                    std::vector<std::string> devices,
                    std::unique_ptr<disc::RalContext> executable)
        : Computation(std::move(computation), std::move(devices)),
          executable(std::move(executable)) {}

    std::unique_ptr<disc::RalContext> executable;
  };
};

}  // namespace runtime
}  // namespace torch_xla
#endif  // XLA_CLIENT_DISC_COMPUTATION_CLIENT_H_
