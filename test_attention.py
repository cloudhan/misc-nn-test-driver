import os
import sys

sys.path.insert(0, os.path.expanduser("~/onnxruntime/build_rocm/Release/build/lib"))


import time

import numpy as np
import onnx
import torch

import onnxruntime as ort

seqlen = 512

input = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT16, ["batchsize", seqlen, 768])
attn_mask = onnx.helper.make_tensor_value_info("attn_mask", onnx.TensorProto.INT32, ["batchsize", seqlen])
output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT16, ["batchsize", seqlen, 768])

np.random.seed(1)
qkv_weight = onnx.helper.make_tensor("qkv_weight", onnx.TensorProto.FLOAT16, [768, 2304], np.random.randn(*[768, 2304]))
qkv_bias = onnx.helper.make_tensor("qkv_bias", onnx.TensorProto.FLOAT16, [2304], np.random.random([2304]))
# qkv_weight = onnx.helper.make_tensor("qkv_weight", onnx.TensorProto.FLOAT16, [768, 2304], np.zeros([768, 2304]))
# qkv_bias = onnx.helper.make_tensor("qkv_bias", onnx.TensorProto.FLOAT16, [2304], np.zeros([2304]))


node = onnx.helper.make_node(
    "Attention",
    inputs=["input", "qkv_weight", "qkv_bias", "attn_mask"],
    outputs=["output"],
    domain="com.microsoft",
    num_heads=12,
)

graph = onnx.helper.make_graph([node], "Attn", [input, attn_mask], [output], initializer=[qkv_weight, qkv_bias])

model = onnx.helper.make_model(
    graph,
    producer_name="tmp",
    opset_imports=[
        onnx.helper.make_opsetid("com.microsoft", 1),
        onnx.helper.make_opsetid("ai.onnx.ml", 1),
        onnx.helper.make_opsetid("", 14),
    ],
)

sess = ort.InferenceSession(
    model.SerializeToString(), providers=[("ROCMExecutionProvider", {"tunable_op_enabled": sys.argv[1]})]
)


input = np.random.randn(64, seqlen, 768)
input = input.astype(np.float16)

inputs = {
    "input": torch.from_numpy(input).cuda(),
    "attn_mask": torch.empty([64, seqlen], dtype=torch.int32).cuda(),
}

outputs = {
    "output": torch.from_numpy(input).cuda(),
}


def create_io_binding(sess, input_tensors, output_tensors):
    def numpy_type(torch_type):
        type_map = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.int32: np.int32,
            torch.int64: np.longlong,
        }
        return type_map[torch_type]

    io_binding = sess.io_binding()
    for name, tensor in input_tensors.items():
        io_binding.bind_input(
            name,
            tensor.device.type,
            0,
            numpy_type(tensor.dtype),
            tensor.shape,
            tensor.data_ptr(),
        )
    for name, tensor in output_tensors.items():
        io_binding.bind_output(
            name,
            tensor.device.type,
            0,
            numpy_type(tensor.dtype),
            tensor.shape,
            tensor.data_ptr(),
        )
    return io_binding


io_binding = create_io_binding(sess, inputs, outputs)

sess.run_with_iobinding(io_binding)

start = time.time()
for i in range(10):
    print(i)
    sess.run_with_iobinding(io_binding)
end = time.time()

print(end - start)
