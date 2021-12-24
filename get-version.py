#!/usr/bin/env python

import argparse
import onnx

parser = argparse.ArgumentParser()
parser.add_argument("onnx_file")
args = parser.parse_args()

model = onnx.load(args.onnx_file)
print(model.ir_version)
