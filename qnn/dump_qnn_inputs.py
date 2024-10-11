import os
import onnx
import numpy as np

def map_tensorproto_type_to_numpy_type(t):
    return {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.BFLOAT16: np.float16,
        onnx.TensorProto.UINT32: np.uint32,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.UINT16: np.uint16,
        onnx.TensorProto.INT16: np.int16,
        onnx.TensorProto.UINT8: np.uint8,
        onnx.TensorProto.INT8: np.int8,
    }[t]


def get_shape_and_dtype(model: onnx.ModelProto, name):
    vi = [i for i in model.graph.input if i.name == name]  # search graph input
    if len(vi) != 0:
        vi = vi[0]
    else:
        vi = [vi for vi in model.graph.value_info if vi.name == name]  # search graph node info
        if len(vi) != 0:
            vi = vi[0]
        else:
            vi = None

    if vi is not None:
        shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
        dtype = map_tensorproto_type_to_numpy_type(vi.type.tensor_type.elem_type)
        return shape, dtype
    else:
        print(f"[ERROR]: cannot find value: {name}")
        return [], np.void


def dump_dummy_inputs(args):
    model = onnx.load_model(args.input_model)
    basename = os.path.basename(args.input_model)
    output_dir = f"{basename}_inputs"
    os.makedirs(output_dir, exist_ok=True)
    for node in model.graph.node:
        if node.op_type != "EPContext":
            continue

        partition_name, = [a.s.decode() for a in node.attribute if a.name == "partition_name"]
        os.makedirs(f"{output_dir}/{partition_name}", exist_ok=True)
        print()
        print(f"[INFO]: Dump context {partition_name} inputs to ./{output_dir}/{partition_name}/")

        input_list_content = ""
        for idx, input_name in enumerate(node.input):
            shape, dtype = get_shape_and_dtype(model, input_name)
            print(f"[INFO]:   input {idx}:", dtype, shape)
            data = np.zeros(shape, dtype=dtype)
            data.tofile(f"{output_dir}/{partition_name}/{idx}.raw")
            input_list_content += (f"{output_dir}/{partition_name}/{idx}.raw ")

        with open(f"{output_dir}/{partition_name}/input_list.txt", "w") as input_list:
            input_list.write(input_list_content.strip())
            input_list.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_model")
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args()

    print(f"[INFO]: Dumping QNN context (dummy) inputs for {args.input_model}")
    dump_dummy_inputs(args)
