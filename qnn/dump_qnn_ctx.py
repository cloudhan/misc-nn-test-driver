import os
import onnxruntime
import onnx
import argparse
import numpy as np


QNN_HOME="/opt/qcom/aistack/qairt/2.26.0.240828"


def create_qnn_session(args):
    so = onnxruntime.SessionOptions()
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "0")
    so.add_session_config_entry("ep.context_enable", "1")
    so.add_session_config_entry("ep.context_embed_mode", "0")

    if args.debug_dump:
        so.add_session_config_entry("session.debug_layout_transformation", "1")

    qnn_options = {
        "backend_path": f"{QNN_HOME}/lib/x86_64-linux-clang/libQnnHtp.so",
        "htp_graph_finalization_optimization_mode": "3",
        "htp_arch": "75",
        # "soc_model": "60",
        "vtcm_mb": "8",
        "enable_htp_fp16_precision": "1",
        "htp_performance_mode": "sustained_high_performance",
    }

    session = onnxruntime.InferenceSession(args.input_model,
                                           sess_options=so,
                                           providers=["QNNExecutionProvider"],
                                           provider_options=[qnn_options])
    return session


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model")
    parser.add_argument("--debug_dump", default=False, action="store_true")

    args = parser.parse_args()

    onnxruntime.set_default_logger_severity(0)

    print(f"[INFO]: Dumping QNN context for {args.input_model}")
    _ = create_qnn_session(args)
