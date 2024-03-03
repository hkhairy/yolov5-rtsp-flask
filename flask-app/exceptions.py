class VideoCapError(Exception):
    def __init__(self, message):
        self.message = "Video capture failed. Check the rtsp url and the network connection"
        self.message += f"\n{message}" if message else ""
        super().__init__(self.message)

