import os
import sys

sys.path.insert(0, os.path.expanduser("~/onnxruntime/build_rocm/Release/build/lib"))


import time

import numpy as np
import onnx
import torch

import onnxruntime as ort

ort.set_default_logger_severity(0)
ort.set_default_logger_verbosity(1000)


def multinormal_distribution(num_distribution, num_element_per_dist):
    arrays = []
    for i in range(num_distribution):
        mean = np.random.randn()
        std = np.random.rand()  # * np.sqrt(num_element_per_dist)
        arrays.append(np.random.normal(mean, std, (num_element_per_dist,)))
    return np.array(arrays)


pack_kv = True
pack_qkv = False
assert int(pack_kv) + int(pack_qkv) == 1

# B = 1
# S = 128
# N = 8
# H = 40

B, S, N, H = 1, 2, 1, 8


np.random.seed(1)
qkv = multinormal_distribution(B * S * N * 3, H).reshape(B, S, N, 3, H)

q_data = qkv[:, :, :, 0, :].reshape(B, S, N * H).astype(np.float16)
kv_data = qkv[:, :, :, 1:, :].astype(np.float16)
qkv_data = qkv.astype(np.float16)

q = onnx.helper.make_tensor_value_info("q", onnx.TensorProto.FLOAT16, ["batchsize", S, N * H])
kv = onnx.helper.make_tensor_value_info("kv", onnx.TensorProto.FLOAT16, ["batchsize", S, N, 2, H])
qkv = onnx.helper.make_tensor_value_info("qkv", onnx.TensorProto.FLOAT16, ["batchsize", S, N, 3, H])

output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT16, ["batchsize", S, N * H])

node_inputs = ["q", "kv"] if pack_kv else ["qkv"]
node = onnx.helper.make_node(
    "MultiHeadAttention", inputs=node_inputs, outputs=["output"], domain="com.microsoft", num_heads=N
)


graph_inputs = [q, kv] if pack_kv else [qkv]
graph = onnx.helper.make_graph([node], "Attn", graph_inputs, [output])

model = onnx.helper.make_model(
    graph,
    producer_name="tmp",
    opset_imports=[
        onnx.helper.make_opsetid("com.microsoft", 1),
        onnx.helper.make_opsetid("ai.onnx.ml", 1),
        onnx.helper.make_opsetid("", 14),
    ],
)

print(onnx.checker.check_model(model))


so = ort.SessionOptions()
so.log_severity_level = 0
so.log_verbosity_level = 1000

sess = ort.InferenceSession(
    model.SerializeToString(),
    providers=[("ROCMExecutionProvider", {"tunable_op_enabled": "1"})],
    sess_options=so,
)

input_feed = {"q": q_data, "kv": kv_data} if pack_kv else {"qkv": qkv_data}

sess.run(output_names=[node.name for node in sess.get_outputs()], input_feed=input_feed)[0]

# def create_io_binding(sess, input_tensors, output_tensors):
#     def numpy_type(torch_type):
#         type_map = {
#             torch.float32: np.float32,
#             torch.float16: np.float16,
#             torch.int32: np.int32,
#             torch.int64: np.longlong,
#         }
#         return type_map[torch_type]

#     io_binding = sess.io_binding()
#     for name, tensor in input_tensors.items():
#         io_binding.bind_input(
#             name,
#             tensor.device.type,
#             0,
#             numpy_type(tensor.dtype),
#             tensor.shape,
#             tensor.data_ptr(),
#         )
#     for name, tensor in output_tensors.items():
#         io_binding.bind_output(
#             name,
#             tensor.device.type,
#             0,
#             numpy_type(tensor.dtype),
#             tensor.shape,
#             tensor.data_ptr(),
#         )
#     return io_binding


# io_binding = create_io_binding(sess, inputs, outputs)

# sess.run_with_iobinding(io_binding)

# start = time.time()
# for i in range(10):
#     print(i)
#     sess.run_with_iobinding(io_binding)
# end = time.time()

# print(end - start)
