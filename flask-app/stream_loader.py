from abc import ABC
import logging
import threading
import time
import numpy as np
from nptyping import NDArray, Shape
import cv2
from exceptions import VideoCapError

logger = logging.getLogger(__name__)

class StreamLoader(ABC):
    def __init__(self, source: str, retry_interval: int):
        raise NotImplementedError("This is an abstract class")

    def load_frame(self)-> NDArray[Shape["*, *, 3"], np.uint8]:
        """Returns a numpy array of the current frame, in RGB format, with the shape (height, width, 3)
        """
        raise NotImplementedError("This is an abstract class")


class RTSPOpenCVStreamLoader(StreamLoader):
    def __init__(self, rtsp_url: str, retry_interval: int):
        self._rtsp_url = rtsp_url
        self._is_running = False
        self._current_frame = None
        self._retry_interval = retry_interval
        self._thread = threading.Thread(
            target = self._update_with_retries, 
            args = [self._retry_interval], 
            daemon=True
        )
        self._thread.start()
    
    def load_frame(self) -> NDArray[Shape["*,*,3"], np.uint8]:
        """Returns a numpy array of the current frame, in RGB format, with the shape (height, width, 3)
        """
        if not self._is_running or self._current_frame is None:
            raise VideoCapError(f"Video capture at {self._rtsp_url} is not running")
        return self._current_frame
    
    def _update_with_retries(self, retry_interval: int = 5):
        """Continuously update the current frame. This method is called in a separate thread

        Args:
            retry_interval (int, optional): Time Interval to wait before retrying. Defaults to 5.
        """
        while True:
            logger.info("Trying to connect to the RTSP stream")
            cap = cv2.VideoCapture(self._rtsp_url)
            self._is_running = cap.isOpened()
            
            while self._is_running:
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to read frame from the RTSP stream")
                    self._is_running = False
                    break
                
                logger.debug("Frame read from RTSP stream")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._current_frame = np.array(frame)
                time.sleep(0.01)
            
            cap.release()
            logger.error("RTSP stream not open. Reconnecting in 5 seconds")
            time.sleep(retry_interval)

