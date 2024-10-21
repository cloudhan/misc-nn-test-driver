#!/bin/bash

QNN_X64_LIB_DIR=/opt/qcom/aistack/qairt/2.26.0.240828/lib/x86_64-linux-clang

export VENDOR_LIB=/vendor/lib64/
export LD_LIBRARY_PATH=${QNN_X64_LIB_DIR}:/vendor/dsp/cdsp:${VENDOR_LIB}
export ADSP_LIBRARY_PATH="${QNN_X64_LIB_DIR}:/vendor/dsp/cdsp:/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp"

export PATH=/home/cloud/android_sdk/ndk/27.1.12297006:${PATH}
export PATH=/home/cloud/android_sdk/ndk/27.1.12297006/toolchains/llvm/prebuilt/linux-x86_64/bin:${PATH}

export PYTHONPATH=/home/cloud/onnxruntime/build_qnn_x64/Release/build/lib:${PYTHONPATH}
export PYTHONPATH=/opt/qcom/aistack/qairt/2.26.0.240828/lib/python/:${PYTHONPATH}

echo "VENDOR_LIB:"
echo $VENDOR_LIB | tr ":" "\n" | nl

echo "LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH | tr ":" "\n" | nl

echo "ADSP_LIBRARY_PATH:"
echo $ADSP_LIBRARY_PATH | tr ":" "\n" | nl

echo "PYTHONPATH:"
echo $PYTHONPATH | tr ":" "\n" | nl

echo "PATH:"
echo $PATH | tr ":" "\n" | nl

$@
