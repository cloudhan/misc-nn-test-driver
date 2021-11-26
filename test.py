import os
import sys
sys.path.insert(0, "/home/guangyunhan/workspaces/onnxruntime/build/Linux/Debug/build/lib/")
os.environ["ORT_DEFAULT_MAX_VLOG_LEVEL"] = "100"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("onnx_file")
parser.add_argument("--repeat", "-r", default=1, type=int)
args = parser.parse_args()

import warnings
import numpy as np
import onnxruntime as ort

ort.set_default_logger_severity(0)

def symbolic_shape_to_concrete_shape(shape, _stoi={}):
    if isinstance(shape, (list, tuple)):
        for name in filter(lambda i: isinstance(i, str), shape):
            while name not in _stoi:
                try:
                    _stoi[name] = int(input(name + ": "))
                except:
                    print("error")
        return [i if isinstance(i, int) else _stoi[i] for i in shape]
    else:
        raise NotImplementedError()


def create_random_inputs(sess):
    onnx_type_to_dtype = {
        "tensor(float)": np.float32,\
    }
    inputs = []
    for i in sess.get_inputs():
        dtype = onnx_type_to_dtype[i.type]
        shape = symbolic_shape_to_concrete_shape(i.shape)
        inputs.append((i.name, np.random.normal(size=shape).astype(dtype)))
    return dict(inputs)

def execute_onnx(filename, inputs=None, provider=None):
    if provider is None:
        providers=["CPUExecutionProvider"]
    else:
        providers=[provider, "CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.optimized_model_filepath = f"{providers[0]}_optimized_{filename}"
    sess = ort.InferenceSession(filename, sess_options=so, providers=providers)
    sess.disable_fallback()
    if inputs is None:
        inputs = create_random_inputs(sess)
    output_names = [node.name for node in sess.get_outputs()]
    results = sess.run(output_names, inputs)
    return results, inputs

cpu_result, inputs = execute_onnx(args.onnx_file)
for i in range(args.repeat):
    cl_result, _ = execute_onnx(args.onnx_file, inputs, provider="OpenCLExecutionProvider")

sum_abs_errors = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i, ref in enumerate(cpu_result):
        print(f"outputs[{i}]  ref:", ref)
        print(f"outputs[{i}]   my:", cl_result[i])
        diff = cl_result[i] - ref
        print(f"outputs[{i}] diff:", diff)
        sum_abs_errors.append(np.sum(np.abs(diff)))

print("=" * 80)
for i, err in enumerate(sum_abs_errors):
    print(f"outputs[{i}]  err:", err)
