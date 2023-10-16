import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image


class EyeClassifier:
    """
    A class for classifying eye states in images and detecting fatigue.
    """

    def __init__(self, model_path='bestModel.h5', base_dir='resources'):
        """
        Initialize the EyeClassifier with model and directory information.

        Args:
            model_path (str): Path to the pre-trained model file.
            base_dir (str): Base directory where cropped images are stored.
        """
        # Constants for eye states
        self.OPEN_LABEL = 'Open_Eyes'
        self.CLOSED_LABEL = 'Closed_Eyes'

        # Path to the directory containing cropped eye images
        self.CROPS_PATH = os.path.join(base_dir, 'crops')

        # Load the pre-trained model
        self.best_model = load_model(model_path)

        # Initialize an empty list for storing preprocessed eye images
        self.crop_images = []

    def preprocess_image(self, img):
        """
        Preprocess an image for model input.

        Args:
            img (PIL.Image.Image): Input image to be preprocessed.

        Returns:
            np.ndarray: Preprocessed image as a NumPy array.
        """
        # Resize the image to 64x64 pixels and convert it to grayscale
        img = img.resize((64, 64)).convert('L')

        # Convert the image to a NumPy array
        img = np.array(img)

        # Normalize the pixel values to be between 0 and 1
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = img / 255.0

        # Add a channel dimension to the image
        img = np.expand_dims(img, -1)

        return img

    def load_images(self):
        """
        Load and preprocess all eye images from a directory.
        """
        # Clear the list of cropped images
        self.crop_images = []

        # Iterate through files in the specified directory
        for filename in os.listdir(self.CROPS_PATH):
            if filename.endswith(".jpg"):
                image_path = os.path.join(self.CROPS_PATH, filename)
                img = Image.open(image_path)
                img = self.preprocess_image(img)
                self.crop_images.append(img)  # Append preprocessed images to the list

        # Convert the list of preprocessed images to a NumPy array
        self.crop_images = np.array(self.crop_images)

    def classify_images(self):
        """
        Classify loaded eye images and visualize the results.

        Returns:
            tuple: A tuple containing counts of open and closed eyes.
        """
        open_count = 0
        close_count = 0

        # Iterate through preprocessed images
        for img in self.crop_images:
            # Use the pre-trained model to make predictions on the image
            result = self.best_model.predict(np.expand_dims(img, 0), verbose=0)

            # Display the image using Matplotlib
            plt.imshow(img.squeeze(), cmap='gray')
            plt.axis('off')
            plt.show()

            if result > 0.5:
                open_count += 1
                # Add a label indicating "Open Eyes" in green
                plt.text(5, 5, self.OPEN_LABEL, color='green', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            else:
                close_count += 1
                # Add a label indicating "Closed Eyes" in red
                plt.text(5, 5, self.CLOSED_LABEL, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        # Return the counts of open and closed eyes
        return
