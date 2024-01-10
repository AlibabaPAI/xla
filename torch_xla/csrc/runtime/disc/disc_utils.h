#ifndef XLA_TORCH_XLA_CSRC_RUNTIME_DISC_DISCUTILS_H_
#define XLA_TORCH_XLA_CSRC_RUNTIME_DISC_DISCUTILS_H_

#include <string>
#include <vector>

namespace torch_xla {
namespace runtime {
namespace disc {
std::vector<char> make_filename(std::string name_prefix);
std::string ReadFileBytes(const std::string& fname);
class TempFile {
 public:
  TempFile(std::string prefix = "");
  ~TempFile();
  TempFile(const TempFile&) = delete;
  void operator=(const TempFile&) = delete;
  /// Write bytes content to temp file and return true on success.
  bool WriteBytesToFile(const std::string& bytes);
  /// Read byte content from temp file.
  std::string ReadBytesFromFile();
  /// Read string content from temp file..
  std::string ReadStringFromFile();
  /// Get the filename of the temp file.
  const std::string& GetFilename() const;

 private:
  std::string fname_;
  int fd_;
};

}  // namespace disc
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_RUNTIME_DISC_DISCUTILS_H_