import numpy as np
import argparse
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize_with_crop_or_pad
import os

def preprocess_data(img_dims, input_path):
    # Load all images from the directory
    images = []
    for filename in os.listdir(input_path):
        if filename.endswith('.png'):
            img_path = os.path.join(input_path, filename)
            img = load_img(img_path, target_size=(img_dims, img_dims))
            img_array = img_to_array(img) / 255.0  # Normalize the image
            images.append(img_array)

    # Convert to numpy array
    images = np.array(images)
    return images

if __name__ == "__main__":
    images = preprocess_data(150, "./img")

    np.save("6.npy", images)
    print(f"Data saved to")
