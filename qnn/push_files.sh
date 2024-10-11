#!/bin/bash

set -ex

adb push --sync                 \
  backend_extension_config.json \
  htp_config.json               \
  convnext_tiny.onnx_ctx.onnx   \
  convnext_tiny.onnx_ctx.onnx_QNNExecutionProvider_QNN_10418545705975462491_1_0.bin \
  convnext_tiny.onnx_ctx.onnx_inputs                                                \
  /data/ort_dev
