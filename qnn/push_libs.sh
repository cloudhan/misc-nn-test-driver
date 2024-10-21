#!/bin/bash

usage() {
    echo "Usage: $0 --libs|-l [ort|qnnhtp]"
    exit 1
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--libs)
            if [[ "$2" != "ort" && "$2" != "qnn" ]]; then
                usage
            fi
            libs=$2
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

export ORT_DIR=$(realpath ~/onnxruntime/build_qnn_android/Release/)
export DEV_DIR=/data/ort_dev
export QNN_HOME=/opt/qcom/aistack/qairt/2.26.0.240828
export NDK_LLVM_DIR=$HOME/android_sdk/ndk/27.1.12297006/toolchains/llvm/prebuilt/linux-x86_64/

set -ex



adb shell mkdir -p ${DEV_DIR}

if [[ $libs == "ort" ]]; then
  tmp=$(mktemp -d)

  ${NDK_LLVM_DIR}/bin/llvm-strip --strip-debug ${ORT_DIR}/libonnxruntime.so -o $tmp/libonnxruntime.so
  ${NDK_LLVM_DIR}/bin/llvm-strip --strip-debug ${ORT_DIR}/onnxruntime_perf_test -o $tmp/onnxruntime_perf_test

  adb push --sync              \
    $tmp/libonnxruntime.so     \
    $tmp/onnxruntime_perf_test \
    ${DEV_DIR}/

  rm -rf $tmp
fi

if [[ $libs == "qnn" ]]; then
  adb push --sync                                                \
    ${QNN_HOME}/lib/aarch64-android/libQnnSystem.so              \
    ${QNN_HOME}/lib/aarch64-android/libQnnSaver.so               \
    ${QNN_HOME}/lib/aarch64-android/libQnnCpu.so                 \
    ${QNN_HOME}/lib/aarch64-android/libQnnGpu.so                 \
    ${QNN_HOME}/lib/aarch64-android/libQnnHtp.so                 \
    ${QNN_HOME}/lib/aarch64-android/libQnnHtpPrepare.so          \
    ${QNN_HOME}/lib/aarch64-android/libQnnHtpV75Stub.so          \
    ${QNN_HOME}/bin/aarch64-android/qnn-net-run                  \
    ${QNN_HOME}/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so     \
    ${QNN_HOME}/lib/aarch64-android/libQnnHtpNetRunExtensions.so \
    backend_extension_config.json                                \
    htp_config.json                                              \
    ${DEV_DIR}/
fi
