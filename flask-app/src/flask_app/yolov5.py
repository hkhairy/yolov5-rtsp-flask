class Preprocessor:
    @staticmethod
    def preprocess(
        img_rgb: NDArray[Shape["*,*,3"], np.uint8],
        canonical_size: int = 640
    ) -> PreProcessingOutput:
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
            PreProcessingOutput: A struct like object with The preprocessed image in BCHW format
                and the Scale factor used to resize the image
        """
        # resize, maintaining aspect ratio
        height, width, c = img_rgb.shape
        max_dim_len = max((height, width))
        # prepare a square zeros image
        square_img = np.zeros((max_dim_len, max_dim_len, c), dtype = np.uint8)
        # assign the image to the square
        square_img[0:height, 0:width, :] = img_rgb
        # resize the square img
        scale_factor = max_dim_len / canonical_size
        interpolation = cv2.INTER_CUBIC if scale_factor > 1 else cv2.INTER_AREA
        square_resized = cv2.resize(square_img, (canonical_size, canonical_size), interpolation = interpolation)

        # change dtype from uint8 to float16
        preprocessed = square_resized.astype(np.float16)

        # change fron 0-255 to 0-1
        preprocessed /= 255

        # expand the 0th dimension to act as batch dim
        preprocessed = np.expand_dims(preprocessed, 0)

        # transpose to BCHW from BHWC
        preprocessed = preprocessed.transpose([0, 3, 1, 2])

        return PreProcessingOutput(preprocessed, scale_factor)

