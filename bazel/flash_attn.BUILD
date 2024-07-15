package(
    default_visibility = [
        "//visibility:public",
    ],
)

cc_library(
    name = "headers",
    hdrs = glob(["csrc/flash_attn/src/*.h"],
                ["csrc/cutlass/include/**/*.h"],),
    includes = ["csrc/cutlass/include"],
    strip_include_prefix = "csrc/flash_attn/src",
)

cc_import(
    name = "flash_attn_cuda",
    hdrs = glob(["csrc/flash_attn/src/*.h"],
                ["csrc/cutlass/include/**/*.h"],),
    shared_library = [":build_flash_attn"][0],
)

genrule(
    name = "build_flash_attn",
    srcs = ["setup.py"],
    outs = ["flash_attn_cuda.so"],
    cmd = ';'.join(['pushd third_party/flash-attention/',
                    'popd',
                    'cp third_party/flash-attention/flash_attn_cuda.so $(OUTS)']),
)
