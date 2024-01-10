#include "torch_xla/csrc/runtime/disc/disc_compile.h"

#include <dlfcn.h>

#include <filesystem>

#include "torch_xla/csrc/runtime/tf_logging.h"

namespace torch_xla {
namespace runtime {
namespace disc {

std::string CurrentLibLocation() {
  Dl_info dl_info;
  dladdr((void*)CurrentLibLocation, &dl_info);
  auto fname = std::string(dl_info.dli_fname);
  return fname.substr(0, fname.find_last_of("/"));
}

std::string CompileCMD(const std::string& mlir_fname,
                       const std::string& out_fname) {
  std::stringstream ss;
  std::string logf = absl::StrCat(mlir_fname, ".log");
  // unset XLA_FLAGS, otherwise tf will throw parse error
  std::string compile_cmd = "unset XLA_FLAGS";
  absl::StrAppend(&compile_cmd, "&&", CurrentLibLocation(),
                  "/disc_compiler_main", " ", mlir_fname, " ", out_fname, " > ",
                  logf, " 2>&1");
  return compile_cmd;
}

std::tuple<std::string, std::string, int> CallDiscCompiler(
    const std::string& mlir_fname) {
  std::string out_fname = mlir_fname + ".out";
  std::string cmd = CompileCMD(mlir_fname, out_fname);
  TF_VLOG(1) << "Executing command: " << cmd << " to compile mhlo...";
  auto ret = std::system(cmd.c_str());
  return {cmd, out_fname, ret};
}

std::shared_ptr<TempFile> DumpMlir(mlir::ModuleOp& stablehlo_module) {
  std::string model_dump_str;
  llvm::raw_string_ostream os(model_dump_str);
  stablehlo_module.print(os);
  os.flush();
  std::shared_ptr<TempFile> stablehlo_file = std::make_shared<TempFile>("mlir");
  stablehlo_file->WriteBytesToFile(model_dump_str);
  return stablehlo_file;
}

DISCComplationResult Compile(mlir::ModuleOp& module,
                             const std::vector<DataMeta>& inputs,
                             const std::vector<DataMeta>& outputs) {
  // Dump stablehlo to file
  DISCComplationResult res;
  auto mlir_file = DumpMlir(module);

  // Compile mhlo
  auto compile_res = CallDiscCompiler(mlir_file->GetFilename());
  auto output_fname = std::get<1>(compile_res);

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