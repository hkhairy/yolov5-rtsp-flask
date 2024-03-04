from typing import Any
import logging
import cv2
import numpy as np
from nptyping import NDArray, Shape
from flask_app.yolov5 import DetectedObject


logger = logging.getLogger(__name__) 

class Annotator:
    """Annotator class
    Takes an image and a list of detected objects and annotates the image with the detected objects
    Inspired by https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-ONNX-Python/main.py
    """

    # The colors to use for the bounding boxes, and the text
    colors_rgb = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
        4: (0, 255, 255),
        5: (255, 0, 255),
        6: (255, 255, 255),
        7: (0, 0, 0),
        8: (128, 128, 128),
        9: (128, 0, 0),
        10: (128, 128, 0),
        11: (0, 255, 0),
        12: (0, 255, 255),
        13: (0, 128, 128),
        14: (0, 0, 128),
        15: (255, 0, 255),
        16: (128, 0, 128),
    }

    def __init__(
        self,
        detected_objects: list[DetectedObject],
        original_image: NDArray[Shape[Any, Any, 3], np.uint8],
        scale_factor: float,
    ):
        self.detected_objects = detected_objects
        self.original_image = original_image
        self.scale_factor = scale_factor

    def annotate(self) -> NDArray[Shape["*,*,3"], np.uint8]:
        """Returns the annotated image

        Returns:
            NDArray[Shape["*,*,3"], np.uint8]: The image after annotation
        """
        annotated_image = self.original_image.copy()
        for detected_object in self.detected_objects:
            x1, y1, x2, y2 = detected_object.box
            x1 = int(x1 * self.scale_factor)
            y1 = int(y1 * self.scale_factor)
            x2 = int(x2 * self.scale_factor)
            y2 = int(y2 * self.scale_factor)

            color = Annotator.colors_rgb[detected_object.class_index % 16]
            annotated_image = cv2.rectangle(
                annotated_image, (x1, y1), (x2, y2), color, 2
            )
            annotated_image = cv2.putText(
                annotated_image,
                detected_object.class_name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
            
        logger.info("Annotated the image")
        return annotated_image
