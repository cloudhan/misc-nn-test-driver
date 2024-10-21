#!/bin/sh

set -ev

ORT_DIR=/data/ort_dev
QNN_LIB_DIR=/data/ort_dev

export VENDOR_LIB=/vendor/lib64/
export LD_LIBRARY_PATH=${ORT_DIR}:${QNN_LIB_DIR}:/vendor/dsp/cdsp:${VENDOR_LIB}
export ADSP_LIBRARY_PATH="${QNN_LIB_DIR}:/vendor/dsp/cdsp:/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp"

# ./onnxruntime_perf_test -m times -r 10 -M -A -I \
#   -p lastrun -e qnn -i 'backend_path|libQnnHtp.so htp_performance_mode|sustained_high_performance enable_htp_fp16_precision|1 qnn_graph_dump_dir|/data/ort_dev enable_qnn_graph_dump|1' \
#   $@

# ./onnxruntime_perf_test -m times -r 10 -M -A -I \
#   -p lastrun -e qnn -i 'backend_path|libQnnHtp.so htp_performance_mode|sustained_high_performance enable_htp_fp16_precision|1' \
#   $@

./onnxruntime_perf_test -m times -r 200 -M -A -I \
  -p lastrun -e qnn -i 'backend_path|libQnnHtp.so htp_performance_mode|sustained_high_performance enable_htp_fp16_precision|0 htp_graph_finalization_optimization_mode|3' \
  $@
