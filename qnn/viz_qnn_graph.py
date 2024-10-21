import json
import onnx
from onnx import TensorProto, helper

DTYPE_MAP = {
    0x8: TensorProto.INT8,
    0x16: TensorProto.INT16,
    0x32: TensorProto.INT32,
    0x64: TensorProto.INT64,
    0x108: TensorProto.UINT8,
    0x116: TensorProto.UINT16,
    0x132: TensorProto.UINT32,
    0x164: TensorProto.UINT64,
    0x216: TensorProto.FLOAT16,
    0x232: TensorProto.FLOAT,
    0x264: TensorProto.DOUBLE,
    0x304: TensorProto.INT4,
    0x308: TensorProto.INT8,
    0x316: TensorProto.INT16,
    0x332: TensorProto.INT32,
    0x404: TensorProto.UINT4,
    0x408: TensorProto.UINT8,
    0x416: TensorProto.UINT16,
    0x432: TensorProto.UINT32,
}

def to_onnx_dtype(dtype):
    return DTYPE_MAP.get(dtype, TensorProto.UNDEFINED)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="result of enable_qnn_graph_dump|1, for example QNNExecutionProvider_QNN_4720831590962694204_1_0.json")

    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    value_infos = []
    nodes = []

    for k, v in data["graph"]["tensors"].items():
        value_infos.append(helper.make_tensor_value_info(k, to_onnx_dtype(v["data_type"]), v["dims"]))

    for k, v in data["graph"]["nodes"].items():
        nodes.append(helper.make_node(op_type=v["type"], inputs=v["input_names"], outputs=v["output_names"], name=k))

    graph_def = helper.make_graph(
        nodes=nodes,
        name="test-model",
        inputs=[],
        outputs=[],
        initializer=[],
        doc_string="test-model",
        value_info=value_infos,
    )

    model = helper.make_model(graph_def, opset_imports=[helper.make_operatorsetid("", 18)])
    onnx.save_model(model, args.json + ".onnx")
