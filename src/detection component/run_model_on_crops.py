from keras.models import load_model
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class EyeClassifier:
    def __init__(self, model_path='bestModel.h5', base_dir='resources'):
        self.OPEN_LABEL = 'Open_Eyes'
        self.CLOSED_LABEL = 'Closed_Eyes'
        self.CROPS_PATH = os.path.join(base_dir, 'crops')
        self.best_model = load_model(model_path)
        self.crop_images = []

    def preprocess_image(self, img):
        img = img.resize((64, 64)).convert('L')
        img = np.array(img)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = img / 255.0
        img = np.expand_dims(img, -1)
        return img

    def load_images(self):
        self.crop_images = []
        for filename in os.listdir(self.CROPS_PATH):
            if filename.endswith(".jpg"):
                image_path = os.path.join(self.CROPS_PATH, filename)
                img = Image.open(image_path)
                img = self.preprocess_image(img)
                self.crop_images.append(img)  # Append images to the list

        # Convert the list to a numpy array after loading all images
        self.crop_images = np.array(self.crop_images)

    def classify_images(self):
        open_count = 0
        close_count = 0
        for img in self.crop_images:
            result = self.best_model.predict(np.expand_dims(img, 0), verbose=0)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.axis('off')
            plt.show()

            if result > 0.5:
                open_count += 1
                plt.text(5, 5, self.OPEN_LABEL, color='green', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            else:
                close_count += 1
                plt.text(5, 5, self.CLOSED_LABEL, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        return open_count, close_count

    def is_fatigue(self):
        self.load_images()
        open_count, close_count = self.classify_images()
        count = open_count + close_count
        if count:
            open_percentage = (open_count / count) * 100
            if open_percentage <= 60:
                print("Person is asleep. Turn on alert.")
                return True
        return False


if __name__ == "__main__":
    eye_classifier = EyeClassifier()
    eye_classifier.is_fatigue()
