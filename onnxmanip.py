from typing import List, Dict, Optional
import onnx
import numpy as np
from onnx import numpy_helper


def load_weight(model: onnx.ModelProto, name: str) -> Optional[np.ndarray]:
  for t in model.graph.initializer:
    if t.name == name:
      return numpy_helper.to_array(t)
  return None


def find_node(model: onnx.ModelProto, node_name: str) -> onnx.NodeProto:
  for n in model.graph.node:
    if n.name == node_name:
      return n
  return None


def find_nodes(model: onnx.ModelProto, node_names: List[str]) -> Dict[str, Optional[onnx.NodeProto]]:
  ret = {}
  for name in node_names:
    ret[name] = None
  for n in model.graph.node:
    if n.name in node_names:
      ret[n.name] = n
  return ret


def remove_unused_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
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
  for i in range(len(model.graph.node) - 1, -1, -1):
    node = model.graph.node[i]
    if node.name not in useful_node_names:
      model.graph.node.pop(i)
      # print("node", i, "removed")
  return model


def create_onnx_replace_outputs(model: onnx.ModelProto, output_file, output_node_names: List[str]):
  output_node_names = set(output_node_names)
  nodes = list(find_nodes(model, output_node_names).values())
  assert None not in nodes
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
