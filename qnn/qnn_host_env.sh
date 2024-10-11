#!/bin/bash

export PATH=/home/cloud/android_sdk/ndk/27.1.12297006:${PATH}
export PATH=/home/cloud/android_sdk/ndk/27.1.12297006/toolchains/llvm/prebuilt/linux-x86_64/bin:${PATH}

export PYTHONPATH=/home/cloud/onnxruntime/build_qnn_x64/Release/build/lib:${PYTHONPATH}
export PYTHONPATH=/opt/qcom/aistack/qairt/2.26.0.240828/lib/python/:${PYTHONPATH}

echo "PYTHONPATH:"
echo $PYTHONPATH | tr ":" "\n" | nl

echo "PATH:"
echo $PATH | tr ":" "\n" | nl

$@
