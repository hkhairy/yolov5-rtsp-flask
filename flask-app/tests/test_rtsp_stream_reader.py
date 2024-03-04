import time
import pytest
import cv2
import numpy as np
from flask_app.stream_loader import RTSPOpenCVStreamLoader
from flask_app.exceptions import VideoCapError
from flask_app.utils import get_config_env_vars

config = get_config_env_vars()

def test_frames_loading_successfully(mocker):
    # Ideally, we would use dependency injection and create a mock class
    # but that would require some significant refactoring of the RTSPOpenCVStreamLoader class
    # I will just settle with mocking

    # dummy frame to be returned from the stream
    mock_frame = np.random.randint(0, 255, (500, 600, 3), dtype=np.uint8)
    
    # mock the cv2 video capture
    mock_video_capture = mocker.patch('cv2.VideoCapture')
    mock_video_capture_instance = mock_video_capture.return_value
    mock_video_capture_instance.isOpened.return_value = True
    mock_video_capture_instance.read.return_value = (True, cv2.cvtColor(mock_frame, cv2.COLOR_RGB2BGR))

    # create the stream loader
    loader = RTSPOpenCVStreamLoader("dummy rtsp url", retry_interval = 0)
    time.sleep(0.5)  # Wait for the thread to start
    
    # the next frame to be returned must match the mock frame
    next_frame = loader.load_frame()
    assert np.array_equal(next_frame, mock_frame)


def test_exception_thrown_when_video_capture_fails(mocker):
    # Ideally, we would use dependency injection and create a mock class
    # but that would require some significant refactoring of the RTSPOpenCVStreamLoader class
    # I will just settle with mocking
    
    # mock the cv2 video capture
    mock_video_capture = mocker.patch('cv2.VideoCapture')
    mock_video_capture_instance = mock_video_capture.return_value
    mock_video_capture_instance.isOpened.return_value = False
    mock_video_capture_instance.read.return_value = (False, None)

    # create the stream loader
    loader = RTSPOpenCVStreamLoader("dummy rtsp url", retry_interval = 0)
    time.sleep(0.5)  # Wait for the thread to start
    
    # an exception must be thrown when trying to load the frame
    with pytest.raises(VideoCapError):
        loader.load_frame()
