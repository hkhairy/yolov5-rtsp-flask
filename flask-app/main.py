import os
import logging
import time
import cv2
from stream_loader import RTSPOpenCVStreamLoader
from exceptions import VideoCapError

# logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

# stream loader setup
stream_loader = RTSPOpenCVStreamLoader(os.getenv("RTSP_URL"), os.getenv("THREAD_RETRY_INTERVAL"))



if __name__ == "__main__":
    stream_loader = RTSPOpenCVStreamLoader(os.getenv("RTSP_URL"), os.getenv("THREAD_RETRY_INTERVAL"))

    while True:
        try:
            frame = stream_loader.load_frame()
            #cv2.imshow("Frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            #cv2.waitKey(0)
            print(frame.shape)
        except VideoCapError:
            logger.error("Failed to read frame from RTSP stream")
            time.sleep(0.5)
            
