#!/bin/sh

ORT_DIR=/data/ort_dev
QNN_LIB_DIR=/data/ort_dev

export VENDOR_LIB=/vendor/lib64/
export LD_LIBRARY_PATH=${ORT_DIR}:${QNN_LIB_DIR}:/vendor/dsp/cdsp:${VENDOR_LIB}
export ADSP_LIBRARY_PATH="${QNN_LIB_DIR}:/vendor/dsp/cdsp:/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp"

echo "VENDOR_LIB:"
echo $VENDOR_LIB | tr ":" "\n" | nl

echo "LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH | tr ":" "\n" | nl

echo "ADSP_LIBRARY_PATH:"
echo $ADSP_LIBRARY_PATH | tr ":" "\n" | nl

$@
