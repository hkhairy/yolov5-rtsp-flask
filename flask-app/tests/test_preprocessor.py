import numpy as np
from flask_app.yolov5 import Preprocessor
import pytest

@pytest.mark.parametrize("width", [500, 1000])
@pytest.mark.parametrize("height", [500, 1000])
def test_preprocessor(width: int, height: int):
    img = np.random.randint(
        low = 0,
        high = 255,
        size = (height, width, 3),
        dtype=np.uint8
    )
    scaled_image, scale_factor = Preprocessor.preprocess(img)

    assert scaled_image.shape == (1, 3, 640, 640)
    assert scaled_image.dtype == np.float16
    assert (scale_factor - max((width, height))/640) < 1e-6

