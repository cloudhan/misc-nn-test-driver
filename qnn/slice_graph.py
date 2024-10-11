import os
import onnx
import argparse
import numpy as np

def map_numpy_type_to_tensorproto_type(t):
    return {
        np.float32: onnx.TensorProto.FLOAT,
        np.float16: onnx.TensorProto.FLOAT16,
        np.float16: onnx.TensorProto.BFLOAT16,
        np.uint32: onnx.TensorProto.UINT32,
        np.int32: onnx.TensorProto.INT32,
        np.uint16: onnx.TensorProto.UINT16,
        np.int16: onnx.TensorProto.INT16,
        np.uint8: onnx.TensorProto.UINT8,
        np.int8: onnx.TensorProto.INT8,
    }[t]


def append_value_info(model, name, shape=None, dtype=None):
    if dtype and shape:
        tp = onnx.helper.make_tensor_type_proto(map_numpy_type_to_tensorproto_type(dtype), shape)
    else:
        tp = onnx.TypeProto()
    vi = onnx.helper.make_value_info(name, tp)
    model.graph.value_info.append(vi)
    return len(model.graph.value_info) - 1


def slice_graph(args):
    model = onnx.load_model(args.input_model)
    # model = onnx.shape_inference.infer_shapes(model)

    basename = os.path.basename(args.input_model)

    inputs = {i.name: i for i in model.graph.input}
    outputs = {o.name: o for o in model.graph.output}
    initializers = {i.name: i for i in model.graph.initializer}
    value_infos = {v.name: v for v in model.graph.value_info}
    tensor_name_to_node = {}
    tensor_name_to_node_index = {}
    for idx, node in enumerate(model.graph.node):
        for tensor_name in node.output:
            tensor_name_to_node_index[tensor_name] = idx
            tensor_name_to_node[tensor_name] = node

    valid_tensor_names = []
    valid_tensor_names.extend(inputs.keys())
    valid_tensor_names.extend(outputs.keys())
    valid_tensor_names.extend(initializers.keys())
    valid_tensor_names.extend(value_infos.keys())
    for i in args.input_tensors:
        # assert i in , f"{i} is not a valid tensor name"
        if i not in valid_tensor_names:
            print(f"[WARNING]: {i} is not a valid tensor name")
            shape = eval(input("shape:"))
            dtype = eval(input("dtype:"))
            # shape = [1,24,28,28]
            # dtype = np.uint8
            idx = append_value_info(model, name=i, shape=shape, dtype=dtype)
            value_infos[i] = model.graph.value_info[idx]
    for o in args.output_tensors:
        # assert o in valid_tensor_names, f"{o} is not a valid tensor name"
        if o not in valid_tensor_names:
            print(f"[WARNING]: {o} is not a valid tensor name")
            idx = append_value_info(model, name=o)
            value_infos[o] = model.graph.value_info[idx]

    pending = [i for i in args.output_tensors]
    visited = set()
    used_inputs = []
    used_outputs = []
    used_initializers = []
    used_node_indices = []
    while len(pending) > 0:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)

        if name in args.output_tensors:
            print(f"[VERBOSE]: use output      {name}")
            used_outputs.append(name)

        if name in args.input_tensors:
            print(f"[VERBOSE]: use input       {name}")
            used_inputs.append(name)
        elif name in initializers:
            print(f"[VERBOSE]: use initializer {name}")
            used_initializers.append(name)
        # elif name in value_infos:  # output from other nodes
        elif name in tensor_name_to_node:  # output from other nodes
            node = tensor_name_to_node[name]
            node_index = tensor_name_to_node_index[name]
            print(f"[VERBOSE]: use node        {node.op_type}({node.name}), node_idx={node_index}")
            used_node_indices.append(node_index)
            pending.extend(node.input)
        else:
            pass
            # print(f"[ERROR]: cannot resolve {name}")
            # assert False

    to_remove = [idx for idx, i in enumerate(model.graph.input) if i.name not in used_inputs]
    for idx in reversed(to_remove):
        model.graph.input.pop(idx)

    to_remove = [idx for idx, o in enumerate(model.graph.output) if o.name not in used_outputs]
    for idx in reversed(to_remove):
        model.graph.output.pop(idx)

    to_remove = [idx for idx, i in enumerate(model.graph.initializer) if i.name not in used_initializers]
    for idx in reversed(to_remove):
        model.graph.initializer.pop(idx)

    to_remove = [idx for idx, _ in enumerate(model.graph.node) if idx not in used_node_indices]
    for idx in reversed(to_remove):
        model.graph.node.pop(idx)

    # for node in model.graph.node:
    #     print(node.input)

    for i in args.input_tensors:
        model.graph.input.append(value_infos[i])

    for o in args.output_tensors:
        model.graph.output.append(value_infos[o])


    onnx.save_model(model, args.input_model + "_sliced.onnx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", help="Input model path")
    parser.add_argument("--input_tensors", "-i", action="append", type=str)
    parser.add_argument("--output_tensors", "-o", action="append", type=str)

    args = parser.parse_args()

    print(f"[INFO]: Slice ONNX Graph {args.input_model} with")
    print(f"[INFO]:   inputs:")
    for i in args.input_tensors:
      print(f"[INFO]:     - {i}")
    print(f"[INFO]:   outputs:")
    for o in args.output_tensors:
      print(f"[INFO]:     - {o}")

    slice_graph(args)
