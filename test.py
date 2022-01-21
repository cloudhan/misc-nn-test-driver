import os
import sys

sys.path.insert(0, "/home/guangyunhan/workspaces/onnxruntime/build/Linux/Release/build/lib/")
os.environ["ORT_DEFAULT_MAX_VLOG_LEVEL"] = "100"
if "LD_PRELOAD" in os.environ and "oclgrind-rt" in os.environ["LD_PRELOAD"]:
  os.environ["LIBOPENCL_SO_PATH"] = os.environ["LD_PRELOAD"]
else:
  os.environ["LIBOPENCL_SO_PATH"] = "/usr/local/cuda-11.1/targets/x86_64-linux/lib/libOpenCL.so"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("onnx_file")
parser.add_argument("--random", action="store_true")
parser.add_argument("--ones", action="store_true", help="generate data with np.ones")
parser.add_argument("--fp16", action="store_true", help="use fp16")
parser.add_argument("--output", action="extend", nargs="+", type=str, default=None)
parser.add_argument("--repeat", "-r", default=1, type=int)
args = parser.parse_args()
print(args)

import warnings
import numpy as np
if not args.random:
  np.random.seed(0)

import tempfile
import onnx
import onnxruntime as ort
from onnxmanip import create_onnx_replace_outputs

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
    if args.ones:
      inputs.append((i.name, np.ones(shape).astype(dtype)))
    else:
      inputs.append((i.name, np.random.normal(size=shape).astype(dtype)))
  return dict(inputs)


def execute_onnx(filename, inputs=None, outputs=None, provider=None):
  if provider is None:
    providers = [("CPUExecutionProvider", {})]
  else:
    providers = [provider, ("CPUExecutionProvider", {})]
  so = ort.SessionOptions()
  so.optimized_model_filepath = f"{providers[0][0]}_optimized_{os.path.basename(filename)}"

  with tempfile.TemporaryDirectory() as tmp_dir:
    if outputs:
      model = onnx.load(filename)
      filename = os.path.join(tmp_dir, os.path.basename(filename))
      create_onnx_replace_outputs(model, filename, outputs)
    sess = ort.InferenceSession(filename, sess_options=so, providers=providers)
    sess.disable_fallback()
    if inputs is None:
      inputs = create_random_inputs(sess)
    output_names = [node.name for node in sess.get_outputs()]
    results = sess.run(output_names, inputs)
    return results, inputs


cpu_result, inputs = execute_onnx(args.onnx_file, outputs=args.output)
for i in range(args.repeat):
  cl_result, _ = execute_onnx(args.onnx_file, inputs, outputs=args.output, provider=("OpenCLExecutionProvider", {"use_fp16": args.fp16}))

abs_errors = []
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  for i, ref in enumerate(cpu_result):
    print(f"outputs[{i}]  ref:", ref)
    print(f"outputs[{i}]   my:", cl_result[i])
    diff = cl_result[i] - ref
    print(f"outputs[{i}] diff:", diff)
    abs_errors.append(np.abs(diff))

print("=" * 80)
for i, err in enumerate(abs_errors):
  print(f"outputs[{i}]  sum abs err:", np.sum(err), "max abs err", np.max(err))
