import time
import cv2

DEFAULT_BASE_DIR: str = 'resources'


class ContinuousPhotoCapture:
    def __init__(self):
        # Initialize the OpenCV video capture object
        self.camera = cv2.VideoCapture(0)
        # Set the resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.running = False

    def capture_photos(self, photos_to_capture):
        captured_count = 0  # Counter for captured photos
        while captured_count < photos_to_capture:
            ret, frame = self.camera.read()
            if not ret:
                continue
            filename = time.strftime(DEFAULT_BASE_DIR + f"/photo_{captured_count}_%Y%m%d%H%M%S.jpg")
            cv2.imwrite(filename, frame)
            captured_count += 1


if __name__ == "__main__":
    capture_instance = ContinuousPhotoCapture()
    capture_instance.capture_photos(photos_to_capture=3)
