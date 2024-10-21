#!/bin/bash

set -ex

export DEV_DIR=/data/ort_dev

adb push --sync perf_test.sh qnn_device_env.sh ${DEV_DIR}/ \

adb push --sync                                                                                    \
  ../../models/convnext_tiny.onnx                                                                  \
  ../../models/convnext_tiny.onnx_ctx.onnx                                                         \
  ../../models/convnext_tiny.onnx_ctx.onnx_QNNExecutionProvider_QNN_10418545705975462491_1_0.bin   \
  ../../models/convnext_tiny.onnx_ctx.onnx_inputs                                                  \
  ../../models/midas_quantized.onnx                                                                \
  ../../models/midas_quantized.onnx_ctx.onnx                                                       \
  ../../models/midas_quantized.onnx_ctx.onnx_QNNExecutionProvider_QNN_3488127089482541571_1_0.bin  \
  ../../models/midas_quantized.onnx_ctx.onnx_inputs                                                \
  ../../models/midas_quantized.onnx_tmp.onnx                                                       \
  ../../models/midas_quantized.onnx_tmp.onnx_ctx.onnx                                              \
  ../../models/midas_quantized.onnx_tmp.onnx_ctx.onnx_inputs                                       \
  ../../models/midas_quantized.onnx_tmp.onnx_ctx.onnx_QNNExecutionProvider_QNN_13089192509860463987_1_0.bin \
  /data/ort_dev
