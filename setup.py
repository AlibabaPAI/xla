#!/usr/bin/env python
# Welcome to the PyTorch/XLA setup.py.
#
# Environment variables you are probably interested in:
#
#   DEBUG
#     build with debug symbols
#
#   TORCH_XLA_VERSION
#     specify the version of PyTorch/XLA, rather than the hard-coded version
#     in this file; used when we're building binaries for distribution
#
#   GIT_VERSIONED_XLA_BUILD
#     creates a git versioned build
#
#   TORCH_XLA_PACKAGE_NAME
#     change the package name to something other than 'torch_xla'
#
#   BAZEL_VERBOSE=0
#     turn on verbose messages during the bazel build of the xla/xrt client
#
#   XLA_CUDA=0
#     build the xla/xrt client with CUDA enabled
#
#   XLA_CPU_USE_ACL=0
#     whether to use ACL
#
#   BUNDLE_LIBTPU=0
#     include libtpu in final wheel

#   BUILD_CPP_TESTS=0
#     build the C++ tests
#
#   GCLOUD_SERVICE_KEY_FILE=''
#     file containing the auth tokens for remote cache/build. implies remote cache.
#
#   BAZEL_REMOTE_CACHE=""
#     whether to use remote cache for builds
#
#   TPUVM_MODE=0
#     whether to build for TPU
#
#   SILO_NAME=""
#     name of the remote build cache silo
#
#   CXX_ABI=""
#     value for cxx_abi flag; if empty, it is inferred from `torch._C`.
#
from setuptools import setup, find_packages, distutils, Extension, command
from setuptools.command import develop, build_ext
import posixpath
import contextlib
import distutils.ccompiler
import distutils.command.clean
import os
import requests
import shutil
import subprocess
import sys
import tempfile
import zipfile

import build_util

base_dir = os.path.dirname(os.path.abspath(__file__))

_date = '20240322'
_libtpu_version = f'0.1.dev{_date}'
_libtpu_storage_path = f'https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/wheels/libtpu-nightly/libtpu_nightly-{_libtpu_version}-py3-none-any.whl'
_jax_version = f'0.4.26.dev{_date}'


def _get_build_mode():
  for i in range(1, len(sys.argv)):
    if not sys.argv[i].startswith('-'):
      return sys.argv[i]


def get_git_head_sha(base_dir):
  xla_git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                        cwd=base_dir).decode('ascii').strip()
  if os.path.isdir(os.path.join(base_dir, '..', '.git')):
    torch_git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                            cwd=os.path.join(
                                                base_dir,
                                                '..')).decode('ascii').strip()
  else:
    torch_git_sha = ''
  return xla_git_sha, torch_git_sha


def get_build_version(xla_git_sha):
  version = os.getenv('TORCH_XLA_VERSION', '2.3.0')
  if build_util.check_env_flag('GIT_VERSIONED_XLA_BUILD', default='TRUE'):
    try:
      version += '+git' + xla_git_sha[:7]
    except Exception:
      pass
  return version


def create_version_files(base_dir, version, xla_git_sha, torch_git_sha):
  print('Building torch_xla version: {}'.format(version))
  print('XLA Commit ID: {}'.format(xla_git_sha))
  print('PyTorch Commit ID: {}'.format(torch_git_sha))
  py_version_path = os.path.join(base_dir, 'torch_xla', 'version.py')
  with open(py_version_path, 'w') as f:
    f.write('# Autogenerated file, do not edit!\n')
    f.write("__version__ = '{}'\n".format(version))
    f.write("__xla_gitrev__ = '{}'\n".format(xla_git_sha))
    f.write("__torch_gitrev__ = '{}'\n".format(torch_git_sha))

  cpp_version_path = os.path.join(base_dir, 'torch_xla', 'csrc', 'version.cpp')
  with open(cpp_version_path, 'w') as f:
    f.write('// Autogenerated file, do not edit!\n')
    f.write('#include "torch_xla/csrc/version.h"\n\n')
    f.write('namespace torch_xla {\n\n')
    f.write('const char XLA_GITREV[] = {{"{}"}};\n'.format(xla_git_sha))
    f.write('const char TORCH_GITREV[] = {{"{}"}};\n\n'.format(torch_git_sha))
    f.write('}  // namespace torch_xla\n')


def maybe_bundle_libtpu(base_dir):
  libtpu_path = os.path.join(base_dir, 'torch_xla', 'lib', 'libtpu.so')
  with contextlib.suppress(FileNotFoundError):
    os.remove(libtpu_path)

  if not build_util.check_env_flag('BUNDLE_LIBTPU', '0'):
    return

  try:
    import libtpu
    module_path = os.path.dirname(libtpu.__file__)
    print('Found pre-installed libtpu at ', module_path)
    shutil.copyfile(os.path.join(module_path, 'libtpu.so'), libtpu_path)
  except ModuleNotFoundError:
    print('No installed libtpu found. Downloading...')

    with tempfile.NamedTemporaryFile('wb') as whl:
      resp = requests.get(_libtpu_storage_path)
      resp.raise_for_status()

      whl.write(resp.content)
      whl.flush()

      os.makedirs(os.path.join(base_dir, 'torch_xla', 'lib'), exist_ok=True)
      with open(libtpu_path, 'wb') as libtpu_so:
        z = zipfile.ZipFile(whl.name)
        libtpu_so.write(z.read('libtpu/libtpu.so'))


class Clean(distutils.command.clean.clean):

  def bazel_clean_(self):
    self.spawn(['bazel', 'clean', '--expunge'])

  def run(self):
    import glob
    import re
    with open('.gitignore', 'r') as f:
      ignores = f.read()
      pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
      for wildcard in filter(None, ignores.split('\n')):
        match = pat.match(wildcard)
        if match:
          if match.group(1):
            # Marker is found and stop reading .gitignore.
            break
          # Ignore lines which begin with '#'.
        else:
          for filename in glob.glob(wildcard):
            try:
              os.remove(filename)
            except OSError:
              shutil.rmtree(filename, ignore_errors=True)

    self.execute(self.bazel_clean_, (), msg="Cleaning bazel outputs")

    # It's an old-style class in Python 2.7...
    distutils.command.clean.clean.run(self)


xla_git_sha, torch_git_sha = get_git_head_sha(base_dir)
version = get_build_version(xla_git_sha)

build_mode = _get_build_mode()
if build_mode not in ['clean']:
  # Generate version info (torch_xla.__version__).
  create_version_files(base_dir, version, xla_git_sha, torch_git_sha)

  # Copy libtpu.so into torch_xla/lib
  maybe_bundle_libtpu(base_dir)


class BazelExtension(Extension):
  """A C/C++ extension that is defined as a Bazel BUILD target."""

  def __init__(self, bazel_target):
    self.bazel_target = bazel_target
    self.relpath, self.target_name = (
        posixpath.relpath(bazel_target, '//').split(':'))
    ext_name = os.path.join(
        self.relpath.replace(posixpath.sep, os.path.sep), self.target_name)
    if ext_name.endswith('.so'):
      ext_name = ext_name[:-3]
    Extension.__init__(self, ext_name, sources=[])


class BuildBazelExtension(build_ext.build_ext):
  """A command that runs Bazel to build a C/C++ extension."""

  def run(self):
    for ext in self.extensions:
      self.bazel_build(ext)
    command.build_ext.build_ext.run(self)

  def bazel_build(self, ext):
    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    bazel_argv = [
        'bazel', 'build', ext.bazel_target,
        f"--symlink_prefix={os.path.join(self.build_temp, 'bazel-')}"
    ]

    import torch
    cxx_abi = os.getenv('CXX_ABI') or getattr(torch._C,
                                              '_GLIBCXX_USE_CXX11_ABI', None)
    if cxx_abi is not None:
      bazel_argv.append(f'--cxxopt=-D_GLIBCXX_USE_CXX11_ABI={int(cxx_abi)}')

    bazel_argv.extend(build_util.bazel_options_from_env())

    self.spawn(bazel_argv)

    ext_bazel_bin_path = os.path.join(self.build_temp, 'bazel-bin', ext.relpath,
                                      ext.target_name)
    ext_dest_path = self.get_ext_fullpath(ext.name)
    ext_dest_dir = os.path.dirname(ext_dest_path)
    if not os.path.exists(ext_dest_dir):
      os.makedirs(ext_dest_dir)
    os.system(f"patchelf --add-rpath '$ORIGIN/' {ext_bazel_bin_path}")
    shutil.copyfile(ext_bazel_bin_path, ext_dest_path)

    # copy flash attention cuda so file
    flash_attn_so_name = 'flash_attn_cuda.so'
    shutil.copyfile(
        '/'.join(['third_party/flash-attention', flash_attn_so_name]),
        '/'.join([ext_dest_dir, flash_attn_so_name]))


class Develop(develop.develop):

  def run(self):
    self.run_command("build_ext")
    super().run()


# Read in README.md for our long_description
cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
  long_description = f.read()

setup(
    name=os.environ.get('TORCH_XLA_PACKAGE_NAME', 'torch_xla'),
    version=version,
    description='XLA bridge for PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/pytorch/xla',
    author='PyTorch/XLA Dev Team',
    author_email='pytorch-xla@googlegroups.com',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8.0",
    packages=find_packages(include=['torch_xla*']),
    ext_modules=[
        BazelExtension('//:_XLAC.so'),
    ],
    install_requires=[
        'absl-py>=1.0.0',
        'cloud-tpu-client>=0.10.0',
        'pyyaml',
        # importlib.metadata backport required for PJRT plugin discovery prior
        # to Python 3.10
        'importlib_metadata>=4.6;python_version<"3.10"',
    ],
    package_data={
        'torch_xla': ['lib/*.so*',],
    },
    entry_points={
        'console_scripts': [
            'stablehlo-to-saved-model = torch_xla.tf_saved_model_integration:main'
        ],
        'torch_xla.plugins': ['tpu = torch_xla._internal.tpu:TpuPlugin',],
    },
    extras_require={
        # On Cloud TPU VM install with:
        # pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
        'tpu': [f'libtpu-nightly=={_libtpu_version}'],
        # pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
        'pallas': [f'jaxlib=={_jax_version}', f'jax=={_jax_version}'],
    },
    cmdclass={
        'build_ext': BuildBazelExtension,
        'clean': Clean,
        'develop': Develop,
    })
