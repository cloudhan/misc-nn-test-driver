import numpy as np
import onnx
from onnx import helper, numpy_helper

def get_new_input_name(inp):
    return ".onnxmanip.insert." + inp

def default_input_processor(model, inp):
    return model

def input_processor(model, inp):
    if "bias" in inp:
        dtype = np.int32
    else:
        dtype = np.int8
    idx = None
    arr = None
    for i, init in enumerate(model.graph.initializer):
        if init.name == inp:
            arr = numpy_helper.to_array(init).copy()
            arr *= 100
            arr = arr.astype(dtype)
            idx = i
            print(f"process {inp}")
            break

    model.graph.initializer.pop(idx)
    model.graph.initializer.insert(idx, numpy_helper.from_array(arr, inp))
    # model.graph.initializer.append(numpy_helper.from_array(arr, inp))

    return model

def subgraph(inp) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    if "bias" in inp:
        dtype = np.int32
    else:
        dtype = np.int8

    scale_name = inp + ".scales"
    zero_point_name = inp + ".zero_points"
    scale_tp = numpy_helper.from_array(np.array(0.01, dtype=np.float32), scale_name)
    zero_point_tp = numpy_helper.from_array(np.array(0, dtype=dtype), zero_point_name)
    dequant = helper.make_node("DequantizeLinear", inputs=[inp, scale_name, zero_point_name], outputs=[get_new_input_name(inp)])
    return [dequant], [scale_tp, zero_point_tp]


def insert_subgraph_before_node_input(model: onnx.ModelProto, node_input: str, subgraph, input_processor=default_input_processor):
    newinp = get_new_input_name(node_input)
    model.graph.value_info.append(helper.make_value_info(newinp, onnx.TypeProto()))

    new_nodes = []
    new_tps = []

    model = input_processor(model, node_input)

    for node_idx, node in enumerate(model.graph.node):
        for idx, inp in enumerate(node.input):
            if inp == node_input:
                # change to new input
                node.input.pop(idx)
                node.input.insert(idx, newinp)

                nodes, tps = subgraph(inp)
                new_nodes.extend([(node_idx, n) for n in nodes])
                new_tps.extend(tps)

    # previously loop in order, append in reverse order before the node to ensure toposort
    for node_idx, node in reversed(new_nodes):
        model.graph.node.insert(node_idx, node)

    inits = set([i.name for i in model.graph.initializer])
    for tp in new_tps:
        if tp.name not in inits:
            inits.add(tp.name)
            model.graph.initializer.append(tp)
    return model


if __name__ == "__main__":
    # output = helper.make_value_info("output", onnx.TypeProto())
    # input = helper.make_value_info("input", onnx.TypeProto())
    # scale_tp = numpy_helper.from_array(np.array(0.01), "scale")
    # zero_point_tp = numpy_helper.from_array(np.array(0, dtype=np.uint8), "zero_point")
    # dequant = helper.make_node("DequantizeLinear", inputs=["input", "scale", "zero_point"], outputs=["output"])
    # helper.make_graph([dequant], "dequant", [input], [output])

    model = onnx.load_model("/mnt/c/Users/guangyunhan/workspaces/models/midas_quantized.onnx")

    model = insert_subgraph_before_node_input(model, "model.pretrained.layer1.0.weight", subgraph, input_processor)
    model = insert_subgraph_before_node_input(model, "model.pretrained.layer1.0.module_conv2d.bias", subgraph, input_processor)
    model = insert_subgraph_before_node_input(model, "model.pretrained.layer1.4.0.conv_dw.weight", subgraph, input_processor)
    model = insert_subgraph_before_node_input(model, "model.pretrained.layer1.4.0.conv_dw.module_conv2d_1.bias", subgraph, input_processor)
    model = insert_subgraph_before_node_input(model, "model.pretrained.layer2.0.0.conv_dw.weight", subgraph, input_processor)
    model = insert_subgraph_before_node_input(model, "model.pretrained.layer2.0.0.conv_dw.module_conv2d_2.bias", subgraph, input_processor)
    model = insert_subgraph_before_node_input(model, "model.pretrained.layer3.0.0.conv_dw.weight", subgraph, input_processor)
    model = insert_subgraph_before_node_input(model, "model.pretrained.layer3.0.0.conv_dw.module_conv2d_3.bias", subgraph, input_processor)
    model = insert_subgraph_before_node_input(model, "model.pretrained.layer4.0.0.conv_dw.weight", subgraph, input_processor)
    model = insert_subgraph_before_node_input(model, "model.pretrained.layer4.0.0.conv_dw.module_conv2d_4.bias", subgraph, input_processor)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save_model(model, "/mnt/c/Users/guangyunhan/workspaces/models/midas_quantized.onnx_tmp.onnx")
