import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

# Set up the image augmentation sequence
# seq = iaa.Sequential([
#     iaa.ColorJitter(brightness=1.5)  # Increase brightness by a factor of 1.5
# ])

seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect color
    # iaa.LinearContrast((0.75, 1.5)), # improve or worsen the contrast
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5) # add gaussian noise to images
])

# Specify the path to the directory containing the images
image_dir = "D:\Retinopathy\RFMiD\Train_Set\demo"
image_dir_2 = "D:\Retinopathy\RFMiD\AugmentedTrain128"

# Get a list of image files in the directory
image_files = os.listdir(image_dir)

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
