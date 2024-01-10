#include "torch_xla/csrc/runtime/disc_computation_client.h"

#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/core/Tensor.h>
#include <torch/cuda.h>

#include <vector>

#include "absl/strings/str_cat.h"
#include "llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"        // from @llvm-project
#include "mlir/IR/BuiltinOps.h"        // from @llvm-project
#include "mlir/IR/MLIRContext.h"       // from @llvm-project
#include "mlir/IR/Operation.h"         // from @llvm-project
#include "mlir/Pass/Pass.h"            // from @llvm-project
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/disc/disc_compile.h"
#include "torch_xla/csrc/runtime/stablehlo_helper.h"
#include "torch_xla/csrc/runtime/sys_util.h"

namespace torch_xla {
namespace runtime {

at::ScalarType TorchTypeFromXlaType(xla::PrimitiveType xla_type) {
  switch (xla_type) {
    case xla::PrimitiveType::BF16:
      return at::ScalarType::BFloat16;
    case xla::PrimitiveType::F16:
      return at::ScalarType::Half;
    case xla::PrimitiveType::F32:
      return at::ScalarType::Float;
    case xla::PrimitiveType::F64:
      return at::ScalarType::Double;
    case xla::PrimitiveType::PRED:
      return at::ScalarType::Bool;
    case xla::PrimitiveType::U8:
      return at::ScalarType::Byte;
    case xla::PrimitiveType::S8:
      return at::ScalarType::Char;
    case xla::PrimitiveType::S16:
    case xla::PrimitiveType::U16:
      return at::ScalarType::Short;
    case xla::PrimitiveType::S32:
    case xla::PrimitiveType::U32:
      return at::ScalarType::Int;
    case xla::PrimitiveType::S64:
    case xla::PrimitiveType::U64:
      return at::ScalarType::Long;
    case xla::PrimitiveType::C64:
      return at::ScalarType::ComplexFloat;
    case xla::PrimitiveType::C128:
      return at::ScalarType::ComplexDouble;
    default:
      XLA_ERROR() << "XLA type not supported: " << xla_type;
  }
}

xla::PrimitiveType XlaTypeFromTorchType(at::ScalarType scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Double:
      return xla::PrimitiveType::F64;
    case at::ScalarType::Float:
      return xla::PrimitiveType::F32;
    case at::ScalarType::BFloat16:
      return xla::PrimitiveType::BF16;
    case at::ScalarType::Half:
      return xla::PrimitiveType::F16;
    case at::ScalarType::Bool:
      return xla::PrimitiveType::PRED;
    case at::ScalarType::Byte:
      return xla::PrimitiveType::U8;
    case at::ScalarType::Char:
      return xla::PrimitiveType::S8;
    case at::ScalarType::Short:
      return xla::PrimitiveType::S16;
    case at::ScalarType::Int:
      return xla::PrimitiveType::S32;
    case at::ScalarType::Long:
      return xla::PrimitiveType::S64;
    case at::ScalarType::ComplexFloat:
      return xla::PrimitiveType::C64;
    case at::ScalarType::ComplexDouble:
      return xla::PrimitiveType::C128;
    default:
      XLA_ERROR() << "Type not supported: " << scalar_type;
  }
}

DISCComputationClient::DISCComputationClient() {
  world_size_ = sys_util::GetEnvInt("WORLD_SIZE", 1);
  local_rank_ = sys_util::GetEnvInt("LOCAL_RANK", 0);
  global_rank_ = sys_util::GetEnvInt("RANK", local_rank_);
}

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

std::vector<ComputationClient::DataPtr> DISCComputationClient::TransferToDevice(
    absl::Span<const std::shared_ptr<const TensorSource>> tensors) {
  std::vector<ComputationClient::DataPtr> datas;
  datas.reserve(tensors.size());

  size_t total_transfered_bytes = 0;

  for (auto& tensor : tensors) {
    std::vector<int64_t> sizes;
    for (auto& dim_val : tensor->shape().dimensions()) {
      sizes.push_back(dim_val);
    }

    auto dtype =
        at::TensorOptions(TorchTypeFromXlaType(tensor->shape().element_type()));
    auto ret = at::empty(sizes, dtype).contiguous();
    // tensor->populate_fn(tensor, ret.data_ptr(),
    //                    ret.element_size() * ret.numel());
    std::memcpy(ret.data_ptr(), tensor->data(),
                ret.element_size() * ret.numel());

    total_transfered_bytes += ret.element_size() * ret.numel();

    if (!torch::cuda::is_available()) {
      XLA_ERROR() << "CUDA is not available.";
    }

    auto device_ret = ret.to(at::kCUDA);
    ComputationClient::DataPtr data = std::make_shared<DISCData>(
        tensor->device(), tensor->shape(), device_ret);
    datas.push_back(data);
  }

  return datas;
}

std::vector<xla::Literal> DISCComputationClient::TransferFromDevice(
    absl::Span<const DataPtr> handles) {
  std::vector<xla::Literal> literals;
  literals.reserve(handles.size());
  for (auto handle : handles) {
    std::shared_ptr<DISCData> disc_data =
        std::dynamic_pointer_cast<DISCData>(handle);
    xla::Shape target_shape =
        xla::ShapeUtil::DeviceShapeToHostShape(xla::ShapeUtil::MakeShape(
            XlaTypeFromTorchType(disc_data->buffer.dtype().toScalarType()),
            disc_data->buffer.sizes()));
    auto& literal = literals.emplace_back(target_shape);
    auto host_data = disc_data->buffer.to(at::kCPU);
    std::memcpy(literal.untyped_data(), host_data.data_ptr(),
                literal.size_bytes());
  }

  return literals;
}

std::vector<ComputationClient::ComputationPtr> DISCComputationClient::Compile(
    std::vector<ComputationClient::CompileInstance> instances) {
  std::vector<ComputationClient::ComputationPtr> computations{};
  for (auto& instance : instances) {
    mlir::MLIRContext context;
    mlir::ModuleOp mlir_module =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    auto status = torch_xla::ConvertHloToMhlo(
        instance.computation.mutable_proto(), &mlir_module);
    XLA_CHECK(status.ok()) << "StableHLO -> MHLO conversion failed.\n"
                           << status.message();

    // Add input and output attributes
    auto entry_func_identifier =
        mlir::StringAttr::get(&context, "tf.entry_function");
    auto input_placement_key =
        mlir::StringAttr::get(&context, "input_placements");
    auto output_placement_key =
        mlir::StringAttr::get(&context, "output_placements");
    auto input_output_alias_params_key =
        mlir::StringAttr::get(&context, "input_output_alias_params");
    auto input_output_alias_outputs_key =
        mlir::StringAttr::get(&context, "input_output_alias_outputs");

    std::string input_placement = "";
    std::string output_placement = "";
    std::string input_output_alias_params = "";
    std::string input_output_alias_outputs = "";

    std::vector<disc::DataMeta> inputs, outputs;

    auto input_output_alias = instance.computation.proto().input_output_alias();
    if (sys_util::GetEnvString("ENBALE_DISC_INPUT_OUTPUT_ALIAS", "") != "OFF") {
      for (const auto& entry : input_output_alias.entries()) {
        input_output_alias_params +=
            std::to_string(entry.parameter_number()) + ",";
        input_output_alias_outputs +=
            std::to_string(entry.output_shape_index(0)) + ",";
      }
    }
    if (!input_output_alias_params.empty()) {
      input_output_alias_params.pop_back();
      input_output_alias_outputs.pop_back();
    }

    // Set attribute for entry function
    mlir::func::FuncOp entry_func;
    for (auto func : mlir_module.getOps<mlir::func::FuncOp>()) {
      if (func.getName().str() == "main") {
        entry_func = func;
        break;
      }
    }

    for (int i = 0; i < entry_func.getFunctionType().getNumInputs(); i++) {
      absl::StrAppend(&input_placement, "gpu,");
      disc::DataMeta tensor_info;
      tensor_info.device = "cuda";
      inputs.push_back(tensor_info);
    }
    if (!input_placement.empty()) {
      input_placement.pop_back();
    }

    if (instance.output_shape->IsTuple()) {
      for (auto& sub_shape : instance.output_shape->tuple_shapes()) {
        absl::StrAppend(&output_placement, "gpu,");
        disc::DataMeta tensor_info;
        tensor_info.device = "cuda";
        tensor_info.scalar_type =
            TorchTypeFromXlaType(sub_shape.element_type());
        outputs.push_back(tensor_info);
      }
    } else {
      absl::StrAppend(&output_placement, "gpu,");
      disc::DataMeta tensor_info;
      tensor_info.device = "cuda";
      tensor_info.scalar_type =
          TorchTypeFromXlaType(instance.output_shape->element_type());
      outputs.push_back(tensor_info);
    }

    if (!output_placement.empty()) {
      output_placement.pop_back();
    }

    auto input_placement_value =
        mlir::StringAttr::get(&context, input_placement);
    auto output_placement_value =
        mlir::StringAttr::get(&context, output_placement);

    auto input_output_alias_outputs_value =
        mlir::StringAttr::get(&context, input_output_alias_outputs);
    auto input_output_alias_params_value =
        mlir::StringAttr::get(&context, input_output_alias_params);

    auto dict = mlir::DictionaryAttr::get(
        &context,
        {mlir::NamedAttribute(input_placement_key, input_placement_value),
         mlir::NamedAttribute(output_placement_key, output_placement_value),
         mlir::NamedAttribute(input_output_alias_params_key,
                              input_output_alias_params_value),
         mlir::NamedAttribute(input_output_alias_outputs_key,
                              input_output_alias_outputs_value)});

    entry_func->setAttr(entry_func_identifier, dict);
    mlir_module->setAttr(entry_func_identifier, dict);

    // Trigger disc compilation
    disc::DISCComplationResult compile_res =
        disc::Compile(mlir_module, inputs, outputs);
    std::shared_ptr<DISCComputation> disc_computation =
        std::make_shared<DISCComputation>(
            std::move(xla::XlaComputation(instance.computation.proto())),
            instance.devices, std::make_unique<disc::RalContext>(compile_res));
    computations.push_back(disc_computation);
  }

  return computations;
}

std::vector<ComputationClient::DataPtr>
DISCComputationClient::ExecuteComputation(
    const ComputationClient::Computation& computation,
    absl::Span<const ComputationClient::DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  const DISCComputation& disc_computation =
      dynamic_cast<const DISCComputation&>(computation);

  std::vector<at::Tensor> buffers;
  buffers.reserve(arguments.size());
  for (auto& argument : arguments) {
    std::shared_ptr<DISCData> disc_data =
        std::dynamic_pointer_cast<DISCData>(argument);
    buffers.push_back(disc_data->buffer);
  }

  std::vector<at::Tensor> results =
      disc_computation.executable->Execute(buffers);

  std::vector<DataPtr> datas;
  datas.reserve(results.size());
  for (auto& result : results) {
    std::shared_ptr<DISCData> data = std::make_shared<DISCData>(
        device, xla::ShapeUtil::MakeShape(xla::F32, result.sizes()), result);

    datas.push_back(data);
  }

  return datas;
}

std::map<std::string, Metric> DISCComputationClient::GetMetrics() const {
  return {};
}

std::string DISCComputationClient::GetDefaultDevice() const {
  return absl::StrCat(DefaultDevicePrefix, std::to_string(local_rank_));
}

std::vector<std::string> DISCComputationClient::GetLocalDevices() const {
  std::vector<std::string> all_devices;
  all_devices.push_back(GetDefaultDevice());
  return all_devices;
}

std::optional<xla::OpSharding> DISCComputationClient::GetDataSharding(
    ComputationClient::DataPtr handle) {
  return std::optional<xla::OpSharding>();
}

void DISCComputationClient::SetReplicationDevices(
    std::shared_ptr<std::vector<std::string>> devices) {
  replication_devices_ = std::move(devices);
}

std::shared_ptr<std::vector<std::string>>
DISCComputationClient::GetReplicationDevices() {
  return replication_devices_;
}

std::vector<std::string> DISCComputationClient::GetAllDevices() const {
  std::vector<std::string> all_devices;
  int device_count = world_size_;
  for (int idx = 0; idx < device_count; idx++) {
    all_devices.push_back(
        absl::StrCat(DefaultDevicePrefix, std::to_string(idx)));
  }
  return all_devices;
}

size_t DISCComputationClient::GetNumDevices() const { return world_size_; }

int DISCComputationClient::GetProcessIndex() const { return local_rank_; }

int DISCComputationClient::GetNumProcesses() const { return world_size_; }

}  // namespace runtime
}  // namespace torch_xla
