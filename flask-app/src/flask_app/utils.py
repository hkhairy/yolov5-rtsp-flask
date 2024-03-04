import os
import logging
from typing import TypedDict


logger = logging.getLogger(__name__)

class ConfigDict(TypedDict):
    RTSP_URL: str|None
    THREAD_RETRY_INTERVAL: float
    IOU_THRESHOLD: float
    OBJECT_DETECTION_SCORE: float
    OUTPUT_DIR: str
    ONNX_LOCAL_FILE_PATH: str
    ONNX_DOWNLOAD_URL: str


def get_config_env_vars() -> ConfigDict:
    """Builds a Configuration dictionary based on the environment variables

    Returns:
        ConfigDict: The configuration dictionary
    """
    config_dict: ConfigDict = {}
    if os.getenv("RTSP_URL") is None:
        logger.warning("RTSP_URL environment variable is not set")
        config_dict["RTSP_URL"] = None
    else:
        config_dict["RTSP_URL"] = os.getenv("RTSP_URL")

    if os.getenv("THREAD_RETRY_INTERVAL") is None:
        logger.warning("THREAD_RETRY_INTERVAL environment variable is not set, defaulting to 5 seconds")
        config_dict["THREAD_RETRY_INTERVAL"] = 5.0
    else:
        config_dict["THREAD_RETRY_INTERVAL"] = float(os.getenv("THREAD_RETRY_INTERVAL"))

    if os.getenv("IOU_THRESHOLD") is None:
        logger.warning("IOU_THRESHOLD environment variable is not set, defaulting to 0.4")
        config_dict["IOU_THRESHOLD"] = 0.4
    else:
        config_dict["IOU_THRESHOLD"] = float(os.getenv("IOU_THRESHOLD"))

    if os.getenv("OBJECT_DETECTION_SCORE") is None:
        logger.warning("OBJECT_DETECTION_SCORE environment variable is not set, defaulting to 0.4")
        config_dict["OBJECT_DETECTION_SCORE"] = 0.4
    else:
        config_dict["OBJECT_DETECTION_SCORE"] = float(os.getenv("OBJECT_DETECTION_SCORE"))

    if os.getenv("OUTPUT_DIR") is None:
        logger.warning("OUTPUT_DIR environment variable is not set, defaulting to ./output")
        config_dict["OUTPUT_DIR"] = "output/"
    else:
        config_dict["OUTPUT_DIR"] = os.getenv("OUTPUT_DIR")
    
    if os.getenv("ONNX_LOCAL_FILE_PATH") is None:
        logger.warning("ONNX_LOCAL_FILE_PATH environment variable is not set, defaulting yolov5.onnx")
        config_dict["ONNX_LOCAL_FILE_PATH"] = "yolov5.onnx"
    else:
        config_dict["ONNX_LOCAL_FILE_PATH"] = os.getenv("ONNX_LOCAL_FILE_PATH")

    if os.getenv("ONNX_DOWNLOAD_URL") is None:
        logger.warning("ONNX DOWNLOAD URL environment variable is not set, defaulting to https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx")
        config_dict["ONNX_DOWNLOAD_URL"] = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx"
    else:
        config_dict["ONNX_DOWNLOAD_URL"] = os.getenv("ONNX_DOWNLOAD_URL")
    
    return config_dict
