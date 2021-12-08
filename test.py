import os
import sys

sys.path.insert(0, "/home/guangyunhan/workspaces/onnxruntime/build/Linux/Debug/build/lib/")
os.environ["ORT_DEFAULT_MAX_VLOG_LEVEL"] = "100"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("onnx_file")
parser.add_argument("--random", action="store_true")
parser.add_argument("--ones", action="store_true", help="generate data with np.ones")
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

def remove_unused_nodes(model):
  out_to_node = {}
  nodename_to_index = {}
  for idx, n in enumerate(model.graph.node):
    for oname in n.output:
      assert oname not in out_to_node
      out_to_node[oname] = n
      nodename_to_index[n.name] = idx
  useful_node_names = []
  w = [out_to_node[o.name] for o in model.graph.output]
  while len(w):
    node = w.pop()
    useful_node_names.append(node.name)
    w.extend([out_to_node[i] for i in node.input if i in out_to_node])
  for i in range(len(model.graph.node)-1, -1, -1):
    node  = model.graph.node[i]
    if node.name not in useful_node_names:
      model.graph.node.pop(i)
      # print("node", i, "removed")
  return model

def creat_onnx_replace_outputs(model, output_file, output_node_names):
  output_node_names = set(output_node_names)
  nodes = []
  for n in model.graph.node:
    if n.name in output_node_names:
      nodes.append(n)
  while len(model.graph.output):
    model.graph.output.pop()
  output_names = []
  for n in nodes:
    for o in n.output:
      output_names.append(o)
  for output_name in set(output_names):
      info = onnx.ValueInfoProto()
      info.name = output_name
      model.graph.output.append(info)

  onnx.save(remove_unused_nodes(model), output_file)

def execute_onnx(filename, inputs=None, outputs=None, provider=None):
  if provider is None:
    providers = ["CPUExecutionProvider"]
  else:
    providers = [provider, "CPUExecutionProvider"]
  so = ort.SessionOptions()
  so.optimized_model_filepath = f"{providers[0]}_optimized_{os.path.basename(filename)}"

  with tempfile.TemporaryDirectory() as tmp_dir:
    if outputs:
      model = onnx.load(filename)
      filename = os.path.join(tmp_dir, os.path.basename(filename))
      creat_onnx_replace_outputs(model, filename, outputs)
    sess = ort.InferenceSession(filename, sess_options=so, providers=providers)
    sess.disable_fallback()
    if inputs is None:
      inputs = create_random_inputs(sess)
    output_names = [node.name for node in sess.get_outputs()]
    results = sess.run(output_names, inputs)
    return results, inputs


cpu_result, inputs = execute_onnx(args.onnx_file, outputs=args.output)
for i in range(args.repeat):
  cl_result, _ = execute_onnx(args.onnx_file, inputs, outputs=args.output, provider="OpenCLExecutionProvider")

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
