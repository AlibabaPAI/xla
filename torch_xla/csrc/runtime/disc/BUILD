load(
    "@tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

load(
    "//bazel:rules_def.bzl",
    "ptxla_cc_library",
    "ptxla_cc_test",
)


ptxla_cc_library(
    name = "disc_ral",
    srcs = [
        "disc_ral.cc",
        "custom_call_flash_attention_forward.cc",
        "custom_call_flash_attention_backward.cc"
    ],
    hdrs = [
        "disc_ral.h",
    ],
    deps = [
        ":disc_utils",
        "@disc_compiler//:disc_ral_cuda",
        "@disc_compiler//:disc_custom_op",
        "@disc_compiler//:headers",
        "@local_config_cuda//cuda:cuda_headers",
        "@torch//:libc10",
        "@torch//:libc10_cuda",
        "@torch//:libtorch_cuda",
        "@flash_attn//:headers",
        "@flash_attn//:flash_attn_cuda",
    ],
    copts = [
        "-DGOOGLE_CUDA",
    ]
)

ptxla_cc_library(
    name = "disc_utils",
    srcs = ["disc_utils.cc"],
    hdrs = [
        "disc_utils.h",
    ],
    deps = [
        "//torch_xla/csrc/runtime:tf_logging",
    ]
)

ptxla_cc_library(
    name = "disc_compile",
    srcs = ["disc_compile.cc"],
    hdrs = [
        "disc_compile.h",
    ],
    deps = [
        ":disc_ral",
        ":disc_utils",
        "//torch_xla/csrc/runtime:tf_logging",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
    copts = [
        "-DGOOGLE_CUDA",
    ]
)

ptxla_cc_test(
    name = "disc_ral_test",
    srcs = ["disc_ral_test.cc"],
    deps = [
        ":disc_ral",
        "@tsl//tsl/platform:test",
        "@tsl//tsl/platform:test_main",
    ]
)
