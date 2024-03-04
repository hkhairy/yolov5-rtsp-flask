import os
import requests
import json
import re
import logging
import numpy as np
from nptyping import NDArray, Shape
import cv2
import onnxruntime as ort


logger = logging.getLogger(__name__)

class DetectedObject:
    """A class to represent a detected object"""

    def __init__(
        self,
        class_index: int,
        class_name: str,
        score: float,
        box: NDArray[Shape["4"], np.int16],
    ):
        self.class_index = class_index
        self.class_name = class_name
        self.score = score
        self.box = box  # (x1, y1) Top left corner,  (x2, y2) Bottom right corner

    def __repr__(self):
        return f"{self.class_name} ({self.class_index}): {self.score:.2f}"


class Preprocessor:
    @staticmethod
    def preprocess(
        img_rgb: NDArray[Shape["*,*,3"], np.uint8], canonical_size: int = 640
    ) -> tuple[NDArray[Shape["1, 3, 640, 640"], np.float16], float]:
        """Preprocess the image to be in a format that the Yolov5 model expects

        1. Create a square image by padding the smaller dimension with zeros
        2. Overlay the original image on the square image
        3. Resize the square image to the canonical size
        4. Convert the image to float16
        5. Change the range from 0-255 to 0-1
        6. Add a batch dimension
        7. Transpose the image from BHWC to BCHW

        Args:
            image (NDArray[Shape["*,*,3"], np.uint8]): an image in RGB format with shape (height, width, 3)
            canonical_size (int, optional): The size to resize the image to. Defaults to 640.

        Returns:
            tuple[NDArray[Shape["1, 3, 640, 640"], np.float16], float]: The preprocessed image and the scale factor
        """
        # resize, maintaining aspect ratio
        height, width, c = img_rgb.shape
        max_dim_len = max((height, width))
        # prepare a square zeros image
        square_img = np.zeros((max_dim_len, max_dim_len, c), dtype=np.uint8)
        # assign the image to the square
        square_img[0:height, 0:width, :] = img_rgb
        # resize the square img
        scale_factor = max_dim_len / canonical_size
        interpolation = cv2.INTER_CUBIC if scale_factor > 1 else cv2.INTER_AREA
        square_resized = cv2.resize(
            square_img, (canonical_size, canonical_size), interpolation=interpolation
        )

        # change dtype from uint8 to float16
        preprocessed = square_resized.astype(np.float16)

        # change fron 0-255 to 0-1
        preprocessed /= 255

        # expand the 0th dimension to act as batch dim
        preprocessed = np.expand_dims(preprocessed, 0)

        # transpose to BCHW from BHWC
        preprocessed = preprocessed.transpose([0, 3, 1, 2])

        logger.info("Preprocessed an image")
        logger.debug(f"Preprocessed image shape: {preprocessed.shape}, scale factor is {scale_factor}")
        return (preprocessed, scale_factor)

class ModelLoader():
    @staticmethod
    def load_model(model_path: str) -> ort.InferenceSession:
        """Load the model from the given path, or download it if it doesn't exist

        Args:
            model_path (str): the local path to the model

        Returns:
            ort.InferenceSession: The Inference Session object
        """
        if os.path.exists(model_path):
            logger.info("Found the model file")
        else:
            logger.warn("Model file not found, trying to download it")
            model_binaries = requests.get("https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx")
            with open(model_path, "wb") as f:
                f.write(model_binaries.content)
            logger.info("Model downloaded")
        return ort.InferenceSession(model_path)


class Model:
    """A class wrapper for the ONNX model and utility functions for the Yolov5 model
    like non-max-suppression, and getting the detected objects
    """

    def __init__(self, model: ort.InferenceSession):
        self.model = model
        self.class_mapping = Model._get_class_index_to_name_mapping(self.model)

    def predict(
        self, preprocessed_img: NDArray[Shape["1, 3, 640, 640"], np.float16]
    ) -> NDArray[Shape["25200,85"], np.float16]:
        """Run the model on the preprocessed image
        Args:
            preprocessed_img (NDArray[Shape["1, 3, 640, 640"], np.float16]): The preprocessed image

        Returns:
            NDArray[25200, 85]: The output of the yolo model, where there 25200 boxes, each with 85 values
            The values are as follows:
            * 4 values for the bounding box (xc, yc, w, h)
            * 1 value for the score
            * 80 values for the class probabilities, based on the COCO dataset
        """
        # we already know it's a single input network
        input_key = self.model.get_inputs()[0].name

        # We don't care about the output names, they will be automatically determined if we set to None
        #   refer to the Session Class in onnx runtime
        # this returns a normal python list of the following shape: [1, 1, 25200, 85]
        output: list = self.model.run(
            output_names=None, input_feed={input_key: preprocessed_img}
        )
        output_np: NDArray[Shape["25200, 85"]] = np.array(output[0][0])

        return output_np

    def non_max_suppression(
        self,
        boxes: NDArray[Shape["25200, 4"], np.float16],
        scores: NDArray[Shape["25200"], np.float16],
        score_threshold: float = 0.4,
        iou_threshold: float = 0.5,
    ) -> NDArray[Shape["*"], np.int16]:
        """Perform non-max suppression on the boxes and scores

        Args:
            boxes (NDArray[Shape["25200, 4"], np.float16]): The bounding boxes (xc, yc, w, h)
            scores (NDArray[Shape["25200"], np.float16]): The scores for whether the box contains an object
            score_threshold (float, optional): The threshold for the score.
            iou_threshold (float, optional): The threshold for the IOU.

        Returns:
            NDArray[Shape["*"], np.int16]: The indices of the boxes to keep
        """
        # convert the boxes to the format that the nms function expects
        boxes_indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, iou_threshold)

        return np.array(boxes_indices, dtype=np.int16)

    def predict_and_get_detected_objects(
        self, preprocessed_img: NDArray[Shape["1,3,640,640"], np.float16]
    ) -> list[DetectedObject]:
        """A convenience function to run the model, do nms, and get the detected objects in one go

        Args:
            img_rgb (NDArray[Shape["1,3,640,640"], np.float16]): The preprocessed image

        Returns:
            list[DetectedObject]: list of detected objects
        """
        yolo_output = self.predict(preprocessed_img)
        boxes_indices = self.non_max_suppression(yolo_output[:, :4], yolo_output[:, 4])
        detected_objects = self.get_detected_objects(yolo_output, boxes_indices)
        return detected_objects

    def get_detected_objects(
        self,
        prediction_output: NDArray[Shape["25200, 85"], np.float16],
        boxes_indices: NDArray[Shape["*"], np.int16],
    ) -> list[DetectedObject]:
        """Get the detected objects from the prediction output and the boxes indices, returned from non-max suppression

        Args:
            prediction_output (NDArray[Shape["25200, 85"], np.float16]):
                The output of the yolo model, where there 25200 boxes, each with 85 values
            boxes_indices (NDArray[Shape["*"], np.int16]):
                The indices of the boxes to keep

        Returns:
            list[DetectedObject]: List of detected objects, duh 😂
        """
        important_outputs: NDArray[Shape["*,85"], np.float16] = prediction_output[
            boxes_indices
        ]
        boxes = important_outputs[:, :4]
        boxes = self.convert_xcyc_to_xyxy(boxes)
        boxes: NDArray[Shape["*, 4"], np.float16] = np.clip(
            boxes, 0, 640
        )  # clip the boxes to the image size

        class_scores = important_outputs[:, 5:]
        predicted_indices = np.argmax(class_scores, axis=1)
        predicted_class_scores = class_scores[:, predicted_indices]
        predicted_classes = [self.get_class_name(index) for index in predicted_indices]

        detected_objects = [
            DetectedObject(class_index, class_name, score, box)
            for class_index, class_name, score, box in zip(
                predicted_indices, predicted_classes, predicted_class_scores, boxes
            )
        ]

        return detected_objects

    def convert_xcyc_to_xyxy(
        self, boxes: NDArray[Shape["*, 4"], np.float16]
    ) -> NDArray[Shape["*, 4"], np.float16]:
        """Convert the boxes from xc, yc, w, h to x1, y1, x2, y2
        where x1,y1 is the topleft corner, x2,y2 is the bottom right corner

        This is also inspired by the yolov5 repo
        https://github.com/ultralytics/yolov5/blob/master/utils/general.py

        Args:
            boxes (NDArray[Shape["*, 4"], np.float16]): The boxes in xc, yc, w, h format

        Returns:
            NDArray[Shape["*, 4"], np.float16]: The boxes in x1, y1, x2, y2 format
        """
        xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1, y1 = xc - w / 2, yc - h / 2
        x2, y2 = xc + w / 2, yc + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    def get_class_name(self, index: int) -> str:
        """Convenience function to get the class name from the class index

        Args:
            index (int): the class index

        Returns:
            str: the class name
        """
        return self.class_mapping[index]

    @staticmethod
    def _get_class_index_to_name_mapping(
        ort_model: ort.InferenceSession,
    ) -> dict[int, str]:
        """loads the dictionary that maps class indices to class names, in a safe way without using `eval`
        This is inspired from the yolov5 repo, in how they're loading onnx models

        Args:
            ort_model (ort.InferenceSession): The ONNXRuntime session model

        Returns:
            dict[int, str]: The dictionary that maps class indices to class names
        """
        meta: dict = ort_model.get_modelmeta().custom_metadata_map
        # the mapping is stored as a string
        class_map_str: str = meta["names"]

        # We will need to convert it to proper json: The keys must be quoted strings, not integers
        class_map = re.sub(
            r"[0-9]+", lambda match: f"'{match.group(0)}'", class_map_str
        )
        # replace single quotes with double quotes
        class_map = class_map.replace("'", '"')
        # load the json as a python dict
        class_map: dict[int, str] = json.loads(class_map)
        # convert the keys to ints for later use
        class_map = {int(key): val for key, val in class_map.items()}
        return class_map
