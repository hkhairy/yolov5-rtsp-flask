import pytest
import numpy as np
from flask_app.yolov5 import Model, Preprocessor, ModelLoader

def test_post_processing_nms():
    """When there are overlapping boxes, the non-max suppression should only return one box
    """
    model = Model(ModelLoader("yolov5s.onnx").model)
    
    xc, yc, w, h, score = 0.5, 0.5, 0.5, 0.5, 0.5
    overlapping_boxes = np.array([[xc, yc, w, h, score]])
    overlapping_boxes = overlapping_boxes.repeat(10, axis = 0)
    print(overlapping_boxes)

    output = model.non_max_suppression(overlapping_boxes[:, :4], overlapping_boxes[:, 4])
    # assert that there's only one box to be detected
    assert len(output) == 1

def test_post_processing_detected_object():
    """given the model output after nms, the detected object should have the correct score, and correct boundaries
    """
    model = Model(ModelLoader.load_model("yolov5s.onnx"))

    xc, yc, w, h, object_detection_score = 1, 1, 1, 1, 0.9
    predicted_class_index = 0
    predicted_class_score = 0.95

    class_scores = np.zeros(80)
    class_scores[predicted_class_index] = predicted_class_score
    full_row = np.append(np.array([xc, yc, w, h, object_detection_score]), class_scores)
    full_row = np.expand_dims(full_row, axis=0)
    

    detected_object = model.get_detected_objects(full_row, boxes_indices=np.array([0]))

    assert detected_object[0].score == predicted_class_score
    assert np.all(detected_object[0].box == np.array((0.5, 0.5, 1.5, 1.5))) # x1, y1, x2, y2

