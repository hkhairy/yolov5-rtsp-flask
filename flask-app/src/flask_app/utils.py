import os
from typing import TypedDict
from flask_app.exceptions import EnvVarNotSet

class ConfigDict(TypedDict):
    RTSP_URL: str
    THREAD_RETRY_INTERVAL: float
    IOU_THRESHOLD: float
    OBJECT_DETECTION_SCORE: float
    OUTPUT_DIR: str
    ONNX_LOCAL_FILE_PATH: str
    ONNX_DOWNLOAD_URL: str


def get_config_env_vars() -> ConfigDict:
    """Builds a Configuration dictionary based on the environment variables

    Raises:
        ValueError: if an environment variable is not set

    Returns:
        ConfigDict: The configuration dictionary
    """
    config_dict: ConfigDict = {}
    if os.getenv("RTSP_URL") is None:
        raise EnvVarNotSet("RTSP_URL environment variable is not set")
    config_dict["RTSP_URL"] = os.getenv("RTSP_URL")

    if os.getenv("THREAD_RETRY_INTERVAL") is None:
        raise EnvVarNotSet("THREAD_RETRY_INTERVAL environment variable is not set")
    config_dict["THREAD_RETRY_INTERVAL"] = float(os.getenv("THREAD_RETRY_INTERVAL"))

    if os.getenv("IOU_THRESHOLD") is None:
        raise EnvVarNotSet("IOU_THRESHOLD environment variable is not set")
    config_dict["IOU_THRESHOLD"] = float(os.getenv("IOU_THRESHOLD"))

    if os.getenv("OBJECT_DETECTION_SCORE") is None:
        raise EnvVarNotSet("OBJECT_DETECTION_SCORE environment variable is not set")
    config_dict["OBJECT_DETECTION_SCORE"] = float(os.getenv("OBJECT_DETECTION_SCORE"))

    if os.getenv("OUTPUT_DIR") is None:
        raise EnvVarNotSet("OUTPUT_DIR environment variable is not set")
    config_dict["OUTPUT_DIR"] = os.getenv("OUTPUT_DIR")
    
    if os.getenv("ONNX_LOCAL_FILE_PATH") is None:
        raise EnvVarNotSet("ONNX_LOCAL_FILE_PATH environment variable is not set")
    config_dict["ONNX_LOCAL_FILE_PATH"] = os.getenv("ONNX_LOCAL_FILE_PATH")

    if os.getenv("ONNX_DOWNLOAD_URL") is None:
        raise EnvVarNotSet("ONNX_DOWNLOAD_URL environment variable is not set")
    config_dict["ONNX_DOWNLOAD_URL"] = os.getenv("ONNX_DOWNLOAD_URL")
    
    return config_dict
