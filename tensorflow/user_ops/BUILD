# Description:
# An example for custom op and kernel defined as a TensorFlow plugin.

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "hard_gate.so",
    srcs = ["hard_gate.cc"],
    gpu_srcs = ["hard_gate.cu.cc"],
#    deps = ["//tensorflow/core:all_kernels"],
    hdrs = ["hard_gate.h",
            "cwise_ops_common.h",
            "cwise_ops_gpu_common.cu.h",
           ],
)

