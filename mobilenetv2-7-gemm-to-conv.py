import onnx
from onnx import helper
from onnx import numpy_helper
from onnxmanip import create_onnx_replace_outputs, find_node, load_weight

model = onnx.load("mobilenetv2-7.onnx")
gap_97 = find_node(model, "GlobalAveragePool_97")
gemm_104 = find_node(model, "Gemm_104")

weight = load_weight(model, "classifier.1.weight")
weight = weight.reshape(weight.shape + (1, 1))
bias = load_weight(model, "classifier.1.bias")

weight = numpy_helper.from_array(weight, name = "conv_output.weight")
bias = numpy_helper.from_array(bias, name = "conv_output.bias")

new_head = helper.make_node("Conv", [gap_97.output[0], weight.name, bias.name], outputs=["new_output"], name="conv_output")

model.graph.node.append(new_head)
model.graph.initializer.append(weight)
model.graph.initializer.append(bias)

create_onnx_replace_outputs(model, "mobilenetv2-7-conv.onnx", ["conv_output"])
