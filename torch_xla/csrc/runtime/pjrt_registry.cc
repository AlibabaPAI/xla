#include "torch_xla/csrc/runtime/pjrt_registry.h"

#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/profiler.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/runtime/torch_allocator.h"
#include "torch_xla/csrc/runtime/xla_coordinator.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/integrations/device_mem_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace torch_xla {
namespace runtime {

namespace {

std::unordered_map<std::string, std::shared_ptr<const PjRtPlugin>>
    pjrt_plugins_;

xla::GpuAllocatorConfig GetGpuAllocatorConfig() {
  auto allocator_config = xla::GpuAllocatorConfig{};
  if (sys_util::GetEnvString(env::kEnvPjrtAllocatorCudaAsync, "").empty() &&
      sys_util::GetEnvString(env::kEnvPjrtAllocatorPreallocate, "").empty() &&
      sys_util::GetEnvString(env::kEnvPjrtAllocatorFraction, "").empty()) {
    return allocator_config;
  }
  if (sys_util::GetEnvBool(env::kEnvPjrtAllocatorCudaAsync, false)) {
    allocator_config.kind = xla::GpuAllocatorConfig::Kind::kCudaAsync;
  }
  allocator_config.preallocate =
      sys_util::GetEnvBool(env::kEnvPjrtAllocatorPreallocate, true);
  allocator_config.memory_fraction =
      sys_util::GetEnvDouble(env::kEnvPjrtAllocatorFraction, 0.75);
  return allocator_config;
}

std::shared_ptr<const PjRtPlugin> GetPjRtPlugin(
    const std::string& device_type) {
  auto plugin_path = pjrt_plugins_.find(device_type);
  return plugin_path != pjrt_plugins_.end() ? plugin_path->second : nullptr;
}

}  // namespace

void RegisterPjRtPlugin(std::string name,
                        std::shared_ptr<const PjRtPlugin> plugin) {
  TF_VLOG(3) << "Registering PjRt plugin " << name;
  pjrt_plugins_[name] = plugin;
}

// Copied from openxla's
// xla/pjrt/gpu/se_gpu_pjrt_client.cc::BuildLocalDeviceStates
absl::StatusOr<std::map<int, std::unique_ptr<xla::LocalDeviceState>>>
BuildLocalDeviceStates(xla::LocalClient* xla_client) {
  std::map<int, std::unique_ptr<xla::LocalDeviceState>> addressable_devices;
  for (stream_executor::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    addressable_devices.emplace(
        executor->device_ordinal(),
        std::make_unique<xla::LocalDeviceState>(
            executor, xla_client, xla::LocalDeviceState::kComputeSynchronized,
            /*max_inflight_computations=*/32,
            /*allow_event_reuse=*/true, /*use_callback_stream=*/true));
  }
  return std::move(addressable_devices);
}

// Modified from openxla's
// xla/pjrt/gpu/se_gpu_pjrt_client.cc::GetStreamExecutorGpuDeviceAllocator
// change to use torch allocator
absl::StatusOr<std::unique_ptr<stream_executor::DeviceMemoryAllocator>>
GetTorchAllocator(stream_executor::Platform* platform,
                  const xla::GpuAllocatorConfig& allocator_config,
                  const std::map<int, std::unique_ptr<xla::LocalDeviceState>>&
                      addressable_devices) {
  std::vector<stream_executor::MultiDeviceAdapter::AllocatorInfo> allocators;
  LOG(INFO) << "Using PyTorch CUDACachingAllocator.";
  for (const auto& ordinal_and_device : addressable_devices) {
    stream_executor::StreamExecutor* executor =
        ordinal_and_device.second->executor();
    int device_ordinal = executor->device_ordinal();
    auto allocator =
        std::make_unique<TorchCUDACachingAllocator>(device_ordinal);
    allocator->SetStreamAndPreallocateMemory(
        ordinal_and_device.second->compute_stream()
            ->platform_specific_handle()
            .stream);
    allocators.emplace_back(std::move(allocator),
                            ordinal_and_device.second->compute_stream(),
                            /*memory_space=*/0);
  }

  // Add any additional allocators for alternate memory spaces.
  for (const auto& ordinal_and_device : addressable_devices) {
    TF_ASSIGN_OR_RETURN(
        auto collective_bfc_allocator,
        xla::CreateCollectiveBFCAllocator(
            ordinal_and_device.second->executor(),
            /*memory_fraction=*/1.0 - allocator_config.memory_fraction,
            allocator_config.collective_memory_size));
    allocators.emplace_back(std::move(collective_bfc_allocator),
                            ordinal_and_device.second->compute_stream(),
                            /*memory_space=*/1);
  }

  for (const auto& ordinal_and_device : addressable_devices) {
    auto host_allocator =
        xla::GetGpuHostAllocator(ordinal_and_device.second->executor());
    allocators.emplace_back(
        std::move(host_allocator), ordinal_and_device.second->compute_stream(),
        /*memory_space=*/
        static_cast<int>(stream_executor::MemoryType::kHost));
  }

  return std::make_unique<stream_executor::MultiDeviceAdapter>(
      platform, std::move(allocators));
}

// Modified from xla::GetStreamExecutorGpuClient, change to use torch allocator
absl::StatusOr<std::unique_ptr<xla::PjRtClient>>
GetPjRtClientWithTorchAllocator(const xla::GpuClientOptions& options) {
  auto pjrt_platform_name = xla::CudaName();

  TF_ASSIGN_OR_RETURN(
      xla::LocalClient * xla_client,
      xla::GetGpuXlaClient(options.platform_name, options.allowed_devices));
  std::map<int, std::unique_ptr<xla::LocalDeviceState>> local_device_states;
  TF_ASSIGN_OR_RETURN(local_device_states, BuildLocalDeviceStates(xla_client));
  xla::EnablePeerAccess(xla_client->backend().stream_executors());

  TF_ASSIGN_OR_RETURN(
      auto allocator,
      GetTorchAllocator(xla_client->platform(), options.allocator_config,
                        local_device_states));

  auto host_memory_allocator =
      xla::GetGpuHostAllocator(local_device_states.begin()->second->executor());

  std::vector<std::unique_ptr<xla::PjRtStreamExecutorDevice>> devices;
  auto gpu_run_options = std::make_unique<xla::gpu::GpuExecutableRunOptions>();
  if (options.enable_mock_nccl) {
    gpu_run_options->set_enable_mock_nccl_collectives();
  }
  std::shared_ptr<xla::KeyValueStoreInterface> kv_store = options.kv_store;
  if (options.enable_mock_nccl) {
    kv_store = std::make_shared<xla::InMemoryKeyValueStore>();
  }
  TF_RET_CHECK(options.num_nodes == 1 || kv_store != nullptr);
  TF_RETURN_IF_ERROR(xla::BuildDistributedDevices(
      pjrt_platform_name, std::move(local_device_states), options.node_id,
      options.num_nodes, &devices, gpu_run_options.get(), kv_store,
      options.enable_mock_nccl));

  return std::unique_ptr<xla::PjRtClient>(
      std::make_unique<xla::StreamExecutorGpuClient>(
          pjrt_platform_name, xla_client, std::move(devices), options.node_id,
          std::move(allocator), std::move(host_memory_allocator),
          options.should_stage_host_to_device_transfers,
          std::move(gpu_run_options)));
}

std::tuple<std::unique_ptr<xla::PjRtClient>, std::unique_ptr<XlaCoordinator>>
InitializePjRt(const std::string& device_type) {
  std::unique_ptr<xla::PjRtClient> client;
  std::unique_ptr<XlaCoordinator> coordinator;

  if (sys_util::GetEnvBool(env::kEnvPjrtDynamicPlugins, false)) {
    std::shared_ptr<const PjRtPlugin> plugin = GetPjRtPlugin(device_type);
    if (plugin) {
      TF_VLOG(1) << "Initializing client for PjRt plugin " << device_type;

      std::shared_ptr<xla::KeyValueStoreInterface> kv_store = nullptr;
      if (plugin->requires_xla_coordinator()) {
        int local_process_rank = sys_util::GetEnvInt(
            env::kEnvPjRtLocalRank, sys_util::GetEnvInt("LOCAL_RANK", 0));
        int global_process_rank =
            sys_util::GetEnvInt("RANK", local_process_rank);
        int local_world_size =
            sys_util::GetEnvInt(env::kEnvPjRtLocalProcessCount,
                                sys_util::GetEnvInt("LOCAL_WORLD_SIZE", 1));
        int global_world_size =
            sys_util::GetEnvInt("WORLD_SIZE", local_world_size);

        std::string master_addr =
            runtime::sys_util::GetEnvString("MASTER_ADDR", "localhost");
        std::string port = runtime::sys_util::GetEnvString(
            "XLA_COORDINATOR_PORT", XlaCoordinator::kDefaultCoordinatorPort);

        TF_VLOG(3) << "Creating coordinator for rank=" << global_process_rank
                   << ", world size=" << global_world_size
                   << ", coordinator address=" << master_addr << ":" << port;

        // Use the XlaCoordinator as the distributed key-value store.
        coordinator = std::make_unique<XlaCoordinator>(
            global_process_rank, global_world_size, master_addr, port);
        std::shared_ptr<xla::DistributedRuntimeClient> distributed_client =
            coordinator->GetClient();
        kv_store = xla::GetDistributedKeyValueStore(distributed_client,
                                                    /*key_prefix=*/"pjrt:");
      }
      const PJRT_Api* c_api = *pjrt::LoadPjrtPlugin(
          absl::AsciiStrToLower(device_type), plugin->library_path());
      XLA_CHECK_OK(pjrt::InitializePjrtPlugin(device_type));
      auto create_options = plugin->client_create_options();
      client = xla::GetCApiClient(
                   absl::AsciiStrToUpper(device_type),
                   {create_options.begin(), create_options.end()}, kv_store)
                   .value();
      profiler::RegisterProfilerForPlugin(c_api);
    }
  } else if (device_type == "CPU") {
    TF_VLOG(1) << "Initializing PjRt CPU client...";
    bool async = sys_util::GetEnvBool(env::kEnvPjrtAsyncCpuClient, true);
    int cpu_device_count = sys_util::GetEnvInt(env::kEnvNumCpu, 1);
    client = std::move(xla::GetTfrtCpuClient(async, cpu_device_count).value());
  } else if (device_type == "TPU") {
    TF_VLOG(1) << "Initializing TFRT TPU client...";
    // Prefer $TPU_LIBRARY_PATH if set
    auto tpu_library_path = sys_util::GetEnvString(
        env::kEnvTpuLibraryPath,
        sys_util::GetEnvString(env::kEnvInferredTpuLibraryPath, "libtpu.so"));
    XLA_CHECK_OK(pjrt::LoadPjrtPlugin("tpu", tpu_library_path).status());
    xla::Status tpu_status = pjrt::InitializePjrtPlugin("tpu");
    XLA_CHECK_OK(tpu_status);
    client = std::move(xla::GetCApiClient("TPU").value());
    const PJRT_Api* c_api =
        static_cast<xla::PjRtCApiClient*>(client.get())->pjrt_c_api();
    profiler::RegisterProfilerForPlugin(c_api);
  } else if (device_type == "TPU_LEGACY") {
    XLA_ERROR() << "TPU_LEGACY client is no longer available.";
  } else if (device_type == "CUDA") {
    TF_VLOG(1) << "Initializing PjRt GPU client...";
    bool async = sys_util::GetEnvBool(env::kEnvPjrtAsyncGpuClient, true);
    int local_process_rank = sys_util::GetEnvInt(
        env::kEnvPjRtLocalRank, sys_util::GetEnvInt("LOCAL_RANK", 0));
    int global_process_rank = sys_util::GetEnvInt("RANK", local_process_rank);
    int local_world_size = sys_util::GetEnvInt("LOCAL_WORLD_SIZE", 1);
    int global_world_size = sys_util::GetEnvInt("WORLD_SIZE", local_world_size);

    TF_VLOG(3) << "Getting StreamExecutorGpuClient for node_id="
               << global_process_rank << ", num_nodes=" << global_world_size
               << ", spmd case=" << sys_util::GetEnvBool("XLA_USE_SPMD", false)
               << ", PJRT_LOCAL_PROCESS_RANK="
               << sys_util::GetEnvString(env::kEnvPjRtLocalRank, "")
               << ", RANK=" << sys_util::GetEnvString("RANK", "")
               << ", LOCAL_WORLD_SIZE="
               << sys_util::GetEnvString("LOCAL_WORLD_SIZE", "")
               << ", WORLD_SIZE=" << sys_util::GetEnvString("WORLD_SIZE", "");
    std::optional<std::set<int>> allowed_devices;
    allowed_devices = std::set{local_process_rank};

    std::shared_ptr<xla::KeyValueStoreInterface> kv_store;
    if (global_world_size > 1) {
      // Use the distributed key-value store from DistributedRuntimeClient.
      std::string master_addr =
          runtime::sys_util::GetEnvString("MASTER_ADDR", "localhost");
      std::string port = runtime::sys_util::GetEnvString(
          "XLA_COORDINATOR_PORT", XlaCoordinator::kDefaultCoordinatorPort);
      coordinator = std::make_unique<XlaCoordinator>(
          global_process_rank, global_world_size, master_addr, port);
      std::shared_ptr<xla::DistributedRuntimeClient> distributed_client =
          coordinator->GetClient();
      kv_store = xla::GetDistributedKeyValueStore(distributed_client,
                                                  /*key_prefix=*/"gpu:");
    }

    xla::GpuClientOptions options;
    options.allocator_config = GetGpuAllocatorConfig();
    options.node_id = global_process_rank;
    options.num_nodes = global_world_size;
    options.allowed_devices = allowed_devices;
    options.platform_name = "gpu";
    options.should_stage_host_to_device_transfers = true;
    options.kv_store = kv_store;
    bool use_torch_allocator =
        sys_util::GetEnvBool(env::kEnvPjrtUseTorchAllocator, false);
    if (use_torch_allocator) {
      client = std::move(GetPjRtClientWithTorchAllocator(options).value());
    } else {
      client = std::move(xla::GetStreamExecutorGpuClient(options).value());
    }
  } else if (device_type == "XPU") {
    TF_VLOG(1) << "Initializing PjRt XPU client...";
    XLA_CHECK_OK(
        pjrt::LoadPjrtPlugin(
            "xpu", sys_util::GetEnvString(env::kEnvXpuLibraryPath, "libxpu.so"))
            .status());
    client = std::move(xla::GetCApiClient("XPU").value());
  } else if (device_type == "NEURON") {
    TF_VLOG(1) << "Initializing PjRt NEURON client...";
    XLA_CHECK_OK(pjrt::LoadPjrtPlugin("NEURON", sys_util::GetEnvString(
                                                    env::kEnvNeuronLibraryPath,
                                                    "libneuronpjrt.so"))
                     .status());
    client = std::move(xla::GetCApiClient("NEURON").value());
  }

  XLA_CHECK(client) << absl::StrFormat("Unknown %s '%s'", env::kEnvPjRtDevice,
                                       device_type);

  return {std::move(client), std::move(coordinator)};
}

}  // namespace runtime
}  // namespace torch_xla
