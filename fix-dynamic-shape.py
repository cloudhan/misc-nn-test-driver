import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("-o", required=True, type=str)
args = parser.parse_args()

import onnx
from onnx.tools import update_model_dims

model = onnx.load(args.model)


symbolics = {}

def gather(values):
    ret = {}
    for i in values:
        dim = []
        for d in i.type.tensor_type.shape.dim:
            if d.dim_param != "":
                dim.append(d.dim_param)
                symbolics[d.dim_param] = float("NaN")
            else:
                dim.append(d.dim_value)
        ret[i.name] = dim
    return ret


def specify(mapping):
    for name in mapping:
        for i, d in enumerate(mapping[name]):
            if isinstance(d, str):
                mapping[name][i] = symbolics[d]

inputs = gather(model.graph.input)
outputs = gather(model.graph.output)
print(inputs, outputs)

for sym in symbolics:
    symbolics[sym] = int(input(sym + ": ").strip())

specify(inputs)
specify(outputs)
print(inputs, outputs)

onnx.save(update_model_dims.update_inputs_outputs_dims(model, inputs, outputs), args.o)
