
package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "headers",
    hdrs = glob(
        [
            "mlir/ral/*.h",
            "mlir/ral/context/base/cuda/*.h",
            "mlir/ral/context/base/cuda/cuda_context_impl.h",
            "mlir/ral/device/cpu/*.h",
            "mlir/ral/device/gpu/*.h",
        ],
    ),
    includes = [
        "tao_compiler",
        "tao_compiler/mlir",
    ],
    strip_include_prefix = "external/disc_compiler/tao_compiler/mlir",
)

cc_import(
    name="disc_ral_cuda",
    shared_library = ":libral_base_context.so",
)

cc_import(
    name="disc_custom_op",
    shared_library = ":libdisc_custom_ops.so",
)

genrule(
    name = "build_disc",
    outs = ["libral_base_context.so", "libdisc_custom_ops.so", "disc_compiler_main", "torch-mlir-opt"],
    local = True,
    cmd = ';'.join(['export PATH=/root/bin:/usr/local/cuda/bin:$${PATH}',
                    'pushd external/disc_compiler/pytorch_blade/',
                    'python ../scripts/python/common_setup.py',
                    'TF_CUDA_COMPUTE_CAPABILITIES="7.0,8.0,8.6,9.0" TORCH_CUDA_ARCH_LIST="7.0 8.0 8.6 9.0" python setup.py bdist_wheel',
                    'popd',
                    'cp third_party/BladeDISC/pytorch_blade/bazel-bin/external/org_disc_compiler/mlir/ral/libral_base_context.so $(location libral_base_context.so)',
                    'cp third_party/BladeDISC/pytorch_blade/bazel-bin/external/org_disc_compiler/mlir/custom_ops/libdisc_custom_ops.so $(location libdisc_custom_ops.so)',
                    'cp third_party/BladeDISC/pytorch_blade/bazel-bin/external/org_disc_compiler/mlir/disc/disc_compiler_main $(location disc_compiler_main)',
                    'cp third_party/BladeDISC/pytorch_blade/bazel-bin/tests/mhlo/torch-mlir-opt/torch-mlir-opt $(location torch-mlir-opt)']),
)
