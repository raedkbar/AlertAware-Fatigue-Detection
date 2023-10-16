import argparse
import os
import cv2
import dlib

from pathlib import Path
from typing import List
from datetime import datetime

DEFAULT_BASE_DIR: str = 'resources'
LABELED_CSV_NAME: str = 'eyes_labeled.csv'
IMAGE_FORMAT: str = "*.jpg"
CROPS_PATH = DEFAULT_BASE_DIR + "/crops"


class DataProcessor:
    """
    Class for processing image data, detecting eyes, and cropping them.
    """

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.image_format = IMAGE_FORMAT
        self.base_dir = DEFAULT_BASE_DIR

    def process_data(self, directory_path):
        """
        Process image data in a directory, detect and crop eyes.

        Args:
            directory_path (str): Path to the directory containing images to process.
        """
        directory_path = Path(directory_path)
        if directory_path.exists():
            file_list: List[Path] = list(directory_path.rglob(IMAGE_FORMAT))

            for i, image in enumerate(file_list):
                image_path: str = image.as_posix()
                self.detect_and_crop_eyes(image_path, i)
                self.delete_original_image(image_path)

    def delete_original_image(self, image_path):
        """
        Delete the original image file.

        Args:
            image_path (str): Path to the original image to delete.
        """
        try:
            os.remove(image_path)
        except Exception as e:
            print(f"Failed to delete original image: {image_path}\nError: {e}")

    def detect_and_crop_eyes(self, image_path, image_index):
        """
        Detect faces and crop eyes in an image.

        Args:
            image_path (str): Path to the image to process.
            image_index (int): Index of the image in the dataset.
        """
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.detector(gray_image)

        for i, face in enumerate(faces):
            landmarks = self.predictor(gray_image, face)

            # Define bounding boxes for both eyes
            eye_boxes = []
            for eye_points in [(36, 39), (42, 45)]:
                left_x = max(0, landmarks.part(eye_points[0]).x - 10)
                top_y = max(0, landmarks.part(eye_points[1]).y - 20)
                right_x = min(image.shape[1], landmarks.part(eye_points[1]).x + 10)
                bottom_y = min(image.shape[0], landmarks.part(eye_points[1]).y + 10)
                eye_boxes.append((left_x, top_y, right_x, bottom_y))

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            for eye_type, (left_x, top_y, right_x, bottom_y) in zip(["left", "right"], eye_boxes):
                if left_x < right_x and top_y < bottom_y:
                    # Crop and resize the eye
                    eye = image[top_y:bottom_y, left_x:right_x]
                    eye_resized = cv2.resize(eye, (64, 64))
                    # Save the cropped and resized eye
                    cv2.imwrite(CROPS_PATH + f"/{eye_type}_eye_{image_index}_{timestamp}.jpg", eye_resized)


def main(argv=None):
    """
    Main function for processing images and cropping eyes.

    Args:
        argv (list): List of command-line arguments.
    """
    parser = argparse.ArgumentParser("Test eyes detection")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    directory_path: Path = Path(args.dir or DEFAULT_BASE_DIR)
    data_processor = DataProcessor()
    data_processor.process_data(directory_path)


if __name__ == "__main__":
    main()
