import os
import logging
import time
import cv2
from flask_app.stream_loader import RTSPOpenCVStreamLoader
from flask_app.exceptions import VideoCapError
from flask_app.yolov5 import Model, Preprocessor
from flask_app.annotator import Annotator
from flask import Flask

# logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

# stream loader setup
#stream_loader = RTSPOpenCVStreamLoader(os.getenv("RTSP_URL"), os.getenv("THREAD_RETRY_INTERVAL"))
stream_loader = RTSPOpenCVStreamLoader("rtsp://localhost:8554/stream", 5)




if __name__ == "__main__":
    
    time.sleep(5) # Wait for the thread to start
    model = Model("yolov5s.onnx")

    while True:
        try:
            frame = stream_loader.load_frame()
            preprocessed_frame, scale_factor = Preprocessor.preprocess(frame)
            detected_objects = model.predict_and_get_detected_objects(preprocessed_frame)

            annotator = Annotator(detected_objects, frame, scale_factor)
            annotated_frame = annotator.annotate()
            
            cv2.imshow("image", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
        except VideoCapError as e:
            logger.error(e)
            
        except Exception as e:
            logger.error(e)
            
