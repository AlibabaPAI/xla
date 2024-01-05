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
    srcs = ["disc_ral.cc"],
    hdrs = [
        "disc_ral.h",
    ],
    deps = [
        "@disc_compiler//:disc_ral_cuda",
        "@disc_compiler//:headers",
        "@local_config_cuda//cuda:cuda_headers",
        "@torch//:libc10",
        "@torch//:libc10_cuda",
        "@torch//:libtorch_cuda",
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
