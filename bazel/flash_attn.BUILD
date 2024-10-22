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
    srcs = glob(["third_party/flash-attention/**"]),
    outs = ["flash_attn_cuda.so"],
    cmd = '&&'.join(['pushd external/flash_attn/',
                    'MAX_JOBS=50 FLASH_ATTENTION_FORCE_BUILD=TRUE python setup.py bdist_wheel',
                    'popd',
                    'cp external/flash_attn/build/*/*.so $(OUTS)']),
    visibility = ["//visibility:public"],
)
