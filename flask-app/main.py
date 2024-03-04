import os
import logging
import datetime
import cv2
from flask_app.stream_loader import RTSPOpenCVStreamLoader
from flask_app.exceptions import VideoCapError
from flask_app.yolov5 import Model, Preprocessor, ModelLoader
from flask_app.annotator import Annotator
from flask_app.utils import get_config_env_vars
from flask import Flask
from flask import request, jsonify

# logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# get the configuration
config = get_config_env_vars()

# set the output directory
os.makedirs(config["OUTPUT_DIR"], exist_ok=True)

# Initialize the Stream Loader
stream_loader = RTSPOpenCVStreamLoader(config["RTSP_URL"], config["THREAD_RETRY_INTERVAL"])

# Load the model
model = Model(
    ModelLoader.load_model(
        config["ONNX_LOCAL_FILE_PATH"],
        config["ONNX_DOWNLOAD_URL"]
    ),
    iou_threshold=config["IOU_THRESHOLD"],
    score_threshold=config["OBJECT_DETECTION_SCORE"]
)

# initialize the flask app
app = Flask(__name__)


@app.route("/object_detection")
def object_detection():
    # validate it's a GET request
    if request.method != "GET":
        return "Only GET requests are supported on this endpoint", 400
    
    # Try to get the next frame from the stream
    try:
        frame = stream_loader.load_frame()
    except VideoCapError as e:
        logger.error(f"Error when trying to load frame {e}")
        return "Internal Server Error: Error when trying to load frame", 500
    except Exception as e:
        return str(e), 500
    
    # Preprocess the frame
    preprocessed_frame, scale_factor = Preprocessor.preprocess(frame)
    # pass it to the model
    detected_objects = model.predict_and_get_detected_objects(preprocessed_frame)
    # annotate a copy of the original frame
    annotated_frame = Annotator(
        detected_objects=detected_objects, 
        original_image=frame, 
        scale_factor=scale_factor
    ).annotate()

    # write the frame to a file
    time_now = datetime.datetime.now().isoformat()
    filename = f"{config['OUTPUT_DIR']}/annotated_frame_{time_now}.jpg"
    # before writing using cv2, we need to convert the color space from RGB to BGR
    cv2.imwrite(filename, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
    return jsonify({"filename" : filename})
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
