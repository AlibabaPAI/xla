#include "torch_xla/csrc/runtime/disc/disc_utils.h"

#include <torch/torch.h>
#include <unistd.h>

#include <fstream>

#include "torch_xla/csrc/runtime/tf_logging.h"

namespace torch_xla {
namespace runtime {
namespace disc {

std::string ReadStringFromEnvVar(const char* env_var_name,
                                 std::string default_val) {
  const char* env_var_val = std::getenv(env_var_name);
  if (env_var_val == nullptr) {
    return default_val;
  }
  return std::string(env_var_val);
}

// This function is copied from c10/util/tempfile.h, so it follows to these
// temperary directory env variables, too.
std::vector<char> make_filename(std::string name_prefix) {
  // The filename argument to `mkstemp` needs "XXXXXX" at the end according to
  // http://pubs.opengroup.org/onlinepubs/009695399/functions/mkstemp.html
  static const std::string kRandomPattern = "XXXXXX";
  // We see if any of these environment variables is set and use their value, or
  // else default the temporary directory to `/tmp`.
  static const char* env_variables[] = {"TMPDIR", "TMP", "TEMP", "TEMPDIR"};
  std::string tmp_directory = "/tmp";
  for (const char* variable : env_variables) {
    auto path = ReadStringFromEnvVar(variable, "");
    if (!path.empty()) {
      tmp_directory = path;
      break;
    }
  }
  std::vector<char> filename;
  filename.reserve(tmp_directory.size() + name_prefix.size() +
                   kRandomPattern.size() + 2);
  filename.insert(filename.end(), tmp_directory.begin(), tmp_directory.end());
  filename.push_back('/');
  filename.insert(filename.end(), name_prefix.begin(), name_prefix.end());
  filename.insert(filename.end(), kRandomPattern.begin(), kRandomPattern.end());
  filename.push_back('\0');
  return filename;
}

std::string ReadFileBytes(const std::string& fname) {
  std::ifstream input(fname, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          (std::istreambuf_iterator<char>()));
  return std::string(bytes.begin(), bytes.end());
}

TempFile::TempFile(std::string prefix) : fname_(""), fd_(-1) {
  auto fname = make_filename(prefix);
  fd_ = mkstemp(fname.data());
  fname_ = std::string(fname.data());
  TORCH_CHECK(fd_ != -1, "Error generating temporary file, file name: ", fname_,
              ", error: ", std::strerror(errno));
}

TempFile::~TempFile() {
  if (!fname_.empty()) {
    ::unlink(fname_.c_str());
  }
  if (fd_ > 0) {
    ::close(fd_);
  }
}

bool TempFile::WriteBytesToFile(const std::string& bytes) {
  ssize_t left_len = bytes.length();
  const char* data = bytes.data();
  errno = 0;
  while (left_len > 0) {
    auto sz = ::write(fd_, data, left_len);
    if (sz <= 0) {
      if (errno != EINTR && errno != EAGAIN) {
        TF_VLOG(1) << "Failed to write content to temp file: " << GetFilename()
                   << ", error: " << strerror(errno);
        return false;
      }
      errno = 0;
      continue;
    }
    left_len -= sz;
    data += sz;
  }
  return true;
}

const std::string& TempFile::GetFilename() const { return fname_; }

std::string TempFile::ReadBytesFromFile() {
  std::ifstream infile(fname_, std::ios::binary);
  std::string str((std::istreambuf_iterator<char>(infile)),
                  std::istreambuf_iterator<char>());
  return str;
}

std::string TempFile::ReadStringFromFile() {
  std::ifstream infile(fname_);
  std::string str((std::istreambuf_iterator<char>(infile)),
                  std::istreambuf_iterator<char>());
  return str;
}

}  // namespace disc
}  // namespace runtime
}  // namespace torch_xla