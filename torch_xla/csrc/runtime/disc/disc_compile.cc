#include "torch_xla/csrc/runtime/disc/disc_compile.h"

#include <dlfcn.h>

#include <filesystem>

#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
using namespace std::filesystem;

namespace torch_xla {
namespace runtime {
namespace disc {

bool IsDiscDebugMode() { return sys_util::GetEnvBool("DISC_DEBUG", false); }

std::string GetDebugDumpDir() {
  return sys_util::GetEnvString("DISC_DEBUG_DUMP_DIR", "./dump_dir");
}

std::string CurrentLibLocation() {
  Dl_info dl_info;
  dladdr((void *)CurrentLibLocation, &dl_info);
  auto fname = std::string(dl_info.dli_fname);
  return fname.substr(0, fname.find_last_of("/"));
}

std::string CompileCMD(const std::string &mlir_fname,
                       const std::string &out_fname) {
  std::stringstream ss;
  std::string logf = absl::StrCat(mlir_fname, ".log");
  // unset XLA_FLAGS, otherwise tf will throw parse error
  std::string compile_cmd = "unset XLA_FLAGS";
  if (IsDiscDebugMode()) {
    absl::StrAppend(&compile_cmd, " && export TF_CPP_VMODULE=disc_compiler=1 ");
  }
  absl::StrAppend(&compile_cmd, "&&", CurrentLibLocation(),
                  "/disc_compiler_main", " ", mlir_fname, " ", out_fname, " > ",
                  logf, " 2>&1");
  return compile_cmd;
}

std::tuple<std::string, std::string, int> CallDiscCompiler(
    const std::string &mlir_fname) {
  std::string out_fname = mlir_fname + ".out";
  std::string cmd = CompileCMD(mlir_fname, out_fname);
  TF_VLOG(1) << "Executing command: " << cmd << " to compile mhlo...";
  auto ret = std::system(cmd.c_str());
  return {cmd, out_fname, ret};
}

std::shared_ptr<TempFile> DumpMlir(mlir::ModuleOp &stablehlo_module) {
  std::string model_dump_str;
  llvm::raw_string_ostream os(model_dump_str);
  stablehlo_module.print(os);
  os.flush();
  std::shared_ptr<TempFile> stablehlo_file = std::make_shared<TempFile>("mlir");
  stablehlo_file->WriteBytesToFile(model_dump_str);
  return stablehlo_file;
}

DISCComplationResult Compile(mlir::ModuleOp &module,
                             const std::vector<DataMeta> &inputs,
                             const std::vector<DataMeta> &outputs) {
  // Dump stablehlo to file
  DISCComplationResult res;
  auto mlir_file = DumpMlir(module);

  // Compile mhlo
  auto compile_res = CallDiscCompiler(mlir_file->GetFilename());
  auto output_fname = std::get<1>(compile_res);

  if (IsDiscDebugMode()) {
    std::string base_path = GetDebugDumpDir();
    auto ret = std::filesystem::create_directory(base_path);
    if (ret != 0) {
      TF_VLOG(0) << "Failed to create dump dir: " << base_path
                 << ", it maybe exists.\n";
    }
    std::string mlir_fname = mlir_file->GetFilename();
    std::string log_fname = absl::StrCat(mlir_fname, ".log");
    std::filesystem::copy_file(
        log_fname,
        absl::StrCat(base_path, "/",
                     std::filesystem::path(mlir_fname).stem().string(),
                     ".log"));
    std::filesystem::copy_file(
        mlir_fname,
        absl::StrCat(base_path, "/",
                     std::filesystem::path(mlir_fname).stem().string(),
                     ".mlir"));
    TF_VLOG(1) << "Dumping mlir to file: " << mlir_file->GetFilename();
  }

  // Construct compiled result
  res.ral_lib = ReadFileBytes(output_fname);
  res.ral_mate_pb = ReadFileBytes(absl::StrCat(output_fname, ".pbtxt"));
  res.inputs = inputs;
  res.outputs = outputs;

  return res;
}

}  // namespace disc
}  // namespace runtime
}  // namespace torch_xla