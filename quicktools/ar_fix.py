import cv2
import os
import numpy as np
from tqdm import tqdm

# Directory containing the images
directory = "./train_data/nitin_crops"

# Iterate over every file in the directory
for filename in tqdm(os.listdir(directory)):
    # Construct the full file path
    filepath = os.path.join(directory, filename)

    # Read the image
    img = cv2.imread(filepath)

    # Calculate the aspect ratio
    try:
        aspect_ratio = img.shape[1] / img.shape[0]
    except:
        print(f"Error reading image: {filepath}")
        continue

    # If the aspect ratio is greater than 6.6666
    if aspect_ratio > 6.6666:
        # Calculate the amount of padding needed
        padding = int((img.shape[1] / 6.6666 - img.shape[0]) / 2)

        # Pad the image's top and bottom with white
        img = cv2.copyMakeBorder(img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # Save the image
        cv2.imwrite(filepath, img)