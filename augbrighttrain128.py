import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import time

# Set up the image augmentation sequence
seq = iaa.Sequential([
    iaa.Multiply((0.75, 1.25)), # Change brightness by a factor between 0.75 and 1.25
])

# Specify the path to the directory containing the images
image_dir = "D:\Retinopathy\RFMiD\ResizedTrain128"
image_dir_2 = "D:\Retinopathy\RFMiD\AugmentedBrightTrain128"

# Get a list of image files in the directory
image_files = os.listdir(image_dir)
num_images = len(image_files)
processed_images = 0

# Start the timer
start_time = time.time()

# Iterate over the image files and perform augmentation
for file_name in image_files:
    # Load the image
    image_path = os.path.join(image_dir, file_name)
    image = cv2.imread(image_path)

    # Apply augmentation to the image
    augmented_images = seq(images=[image])


    # Save augmented images
    for i, augmented_image in enumerate(augmented_images):
        output_file_name = f"augmented_{i}_{file_name}"
        output_path = os.path.join(image_dir_2, output_file_name)
        cv2.imwrite(output_path, augmented_image)
    
    processed_images += 1

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Calculate the estimated time left
    time_per_image = elapsed_time / processed_images
    time_left = (num_images - processed_images) * time_per_image

    # Print the progress and estimated time left
    print(f"Processed: {processed_images}/{num_images} | Time Left: {time_left:.2f} seconds")

# Print the total execution time
total_time = time.time() - start_time
print(f"Total Execution Time: {total_time:.2f} seconds")
