import argparse
import json
import os

import numpy as np

import onnxruntime
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize,
    quantize_static,
)
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
from onnxruntime.quantization.shape_inference import quant_pre_process

print("onnxruntime build = ", onnxruntime.get_build_info())


class RandomDataReader(CalibrationDataReader):
    def __init__(self, scale, offset, model_path, input_name, workload):
        super().__init__()
        self.counter = 0
        self.scale = scale
        self.offset = offset
        self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = input_name
        self.workload = workload

    def get_next(self):
        assert len(self.session.get_inputs()) == 1
        if self.counter >= 300:
            return None

        inp = self.session.get_inputs()[0]
        inputs = np.random.random(inp.shape).astype(np.float32)
        self.counter += 1

        return {self.input_name: inputs}


QUANT_TYPES = {
    "uint8": QuantType.QUInt8,
    "int8": QuantType.QInt8,
    "uint16": QuantType.QUInt16,
    "int16": QuantType.QInt16,
}
CALIBRATION_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "percentile": CalibrationMethod.Percentile,
    "entropy": CalibrationMethod.Entropy,
    "distribution": CalibrationMethod.Distribution,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Quantizes models")
    parser.add_argument(
        "--use_ort_settings",
        required=False,
        action="store_true",
        default=False,
        help="Use ORT's quantization recipe",
    )
    parser.add_argument(
        "--activation_type",
        required=False,
        default="uint8",
        choices=list(QUANT_TYPES.keys()),
        help="Quantized type for activations",
    )
    parser.add_argument(
        "--weight_type",
        required=False,
        default="uint8",
        choices=list(QUANT_TYPES.keys()),
        help="Quantized type for weights",
    )
    parser.add_argument(
        "--per_channel",
        action="store_true",
        default=False,
        help="Enable per-channel weight quantization (defaults to False for per-tensor)",
    )
    parser.add_argument(
        "--calibration_method",
        required=False,
        default="minmax",
        choices=list(CALIBRATION_METHODS.keys()),
        help="Model calibration method",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    per_channel = args.per_channel
    activation_type = QUANT_TYPES[args.activation_type]
    weight_type = QUANT_TYPES[args.weight_type]
    calibration_method = CALIBRATION_METHODS[args.calibration_method]

    models = [
        # {
        #     "name": "text-classification",
        #     "path": "f32_onnx_models/bert_tiny_f32.onnx",
        #     "dataset": "calibration_input_data/Text_Classification_500.json",
        #     "scale": 1,
        #     "offset": 0,
        #     "input_name": "serving_default_input_ids:0",
        # },
        # {
        #     "name": "face-detection",
        #     "path": "f32_onnx_models/retinaface_f32.onnx",
        #     "dataset": "calibration_input_data/Face_Detection_300.json",
        #     "scale": 1,
        #     "offset": 0,
        #     "input_name": "input_1",
        # },
        # {
        #     "name": "image-super-resolution",
        #     "path": "f32_onnx_models/rfdn_f32.onnx",
        #     "dataset": "calibration_input_data/Super_Resolution_500.json",
        #     "scale": 1.0 / 255.0,
        #     "offset": 0,
        #     "input_name": "serving_default_input_4:0",
        # },
        # {
        #     "name": "pose-estimation",
        #     "path": "f32_onnx_models/openposev2_vgg19_f32.onnx",
        #     "dataset": "calibration_input_data/Pose_Estimation_500.json",
        #     "scale": 1,
        #     "offset": 0,
        #     "input_name": "serving_default_input:0",
        # },
        # {
        #     "name": "style-transfer",
        #     "path": "f32_onnx_models/imagetransformnet_f32.onnx",
        #     "dataset": "calibration_input_data/Style_Transfer_500.json",
        #     "scale": 1.0 / 255.0,
        #     "offset": 0,
        #     "input_name": "serving_default_inputs:0",
        # },
        # {
        #     "name": "image-segmentation",
        #     "path": "f32_onnx_models/deeplabv3_mobilenetv2_f32.onnx",
        #     "dataset": "calibration_input_data/Image_Segmentation_500.json",
        #     "scale": 1,
        #     "offset": 0,
        #     "input_name": "input_1",
        # },
        # {
        #     "name": "depth-estimation",
        #     "path": "f32_onnx_models/de_efficientnetlitev3_f32.onnx",
        #     "dataset": "calibration_input_data/Depth_Estimation_500.json",
        #     "scale": 1,
        #     "offset": 0,
        #     "input_name": "input_2",
        # },
        # {
        #     "name": "image-classification",
        #     "path": "f32_onnx_models/mobilenet_v1_f32.onnx",
        #     "dataset": "calibration_input_data/Image_Classification_500.json",
        #     "scale": 1,
        #     "offset": 0,
        #     "input_name": "input_1",
        # },
        # {
        #     "name": "object-detection",
        #     "path": "f32_onnx_models/mobilenetv1_ssd_f32.onnx",
        #     "dataset": "calibration_input_data/Object_Detection_180.json",  # Note: original PrimateLabs scripts use Object_Detection_500.json, but that was not provided.
        #     "scale": 2,
        #     "offset": -1,
        #     "input_name": "serving_default_input_3:0",
        # },
        {
            "name": "midas",
            "path": "f32_onnx_models/midas.onnx",
            "dataset": "calibration_input_data/Depth_Estimation_500.json",
            "scale": 1,
            "offset": 0,
            "input_name": "image",
        },
    ]

    for f32_model_info in models:
        # First preprocessing pass for f32 model.
        print(f"\n[INFO] Preprocessing {f32_model_info['path']}")
        f32_model_path_1 = f32_model_info["path"].replace(".onnx", ".pp1.onnx")
        quant_pre_process(f32_model_info["path"], f32_model_path_1)

        if args.use_ort_settings:
            # Second preprocessing pass for f32 model to apply fusions needed by QNN EP.
            print(f"[INFO] Preprocessing {f32_model_path_1}")
            f32_model_path_2 = f32_model_path_1.replace(".onnx", ".pp2.onnx")
            did_preprocess: bool = qnn_preprocess_model(f32_model_path_1, f32_model_path_2)
            model_to_quantize_path = f32_model_path_2 if did_preprocess else f32_model_path_1
        else:
            model_to_quantize_path = f32_model_path_1

        # Load input calibration data from JSON files provided by PrimateLabs.
        print(f"[INFO] Loading input calibration data from {f32_model_info['dataset']}")
        # with open(f32_model_info["dataset"]) as fd:
        #     dataset = np.array(json.load(fd)).astype(np.float32)\

        # Create a "CalibrationDataReader" that will provide input data during model quantization.
        # calibration_data_provider = CalibrationDataProvider(f32_model_info["scale"],
        #                                                     f32_model_info["offset"],
        #                                                     dataset,
        #                                                     f32_model_info["input_name"],
        #                                                     f32_model_info["name"])

        calibration_data_provider = RandomDataReader(
            f32_model_info["scale"],
            f32_model_info["offset"],
            model_to_quantize_path,
            f32_model_info["input_name"],
            f32_model_info["name"],
        )

        if args.use_ort_settings:
            print(f"[INFO] Quantizing {model_to_quantize_path} with settings:")
            print(f"\tUsing ONNX Runtime's quantization recipe")
            print(f"\t{per_channel=}")
            print(f"\tcalibration_method={args.calibration_method}")
            print(f"\tactivation_type={args.activation_type}")
            print(f"\tweight_type={args.weight_type}")
            per_chan_label = "perchannel" if per_channel else "pertensor"
            output_model_path = f"quantized_models/{f32_model_info['name']}.ORT.{per_chan_label}.{args.calibration_method}.A{args.activation_type}.W{args.weight_type}.qdq.onnx"
            quant_config = get_qnn_qdq_config(
                model_to_quantize_path,
                calibration_data_provider,
                per_channel=per_channel,
                calibrate_method=calibration_method,
                activation_type=activation_type,
                weight_type=weight_type,
            )

            quantize(model_to_quantize_path, output_model_path, quant_config)
        else:
            per_channel = f32_model_info["name"] != "depth-estimation"

            print(f"[INFO] Quantizing {model_to_quantize_path} with settings:")
            print(f"\tUsing PrimateLab's original quantization recipe")
            print(f"\t{per_channel=}")
            print(f"\tcalibration_method=minmax")
            print(f"\tactivation_type=uint8")
            print(f"\tweight_type=uint8")
            per_chan_label = "perchannel" if per_channel else "pertensor"
            output_model_path = (
                f"quantized_models/{f32_model_info['name']}.PL.{per_chan_label}.minmax.Auint8.Wuint8.qdq.onnx"
            )
            # Original quantization call used by PrimateLabs.
            quantize_static(
                model_to_quantize_path,
                output_model_path,
                calibration_data_provider,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QUInt8,
                per_channel=per_channel,
            )

        print(f"[INFO] Generated quantized model: {output_model_path}")
        # break  # TODO: Remove to quantize all models


if __name__ == "__main__":
    main()
