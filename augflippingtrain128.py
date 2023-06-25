import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

# Set up the image augmentation sequence
seq = iaa.Sequential([
    iaa.Flipud(p=1.0)
    # iaa.Fliplr(p=1.0)
])

# Specify the path to the directory containing the images
image_dir = "D:\Retinopathy\RFMiD\AugmentedHueTrain128"
image_dir_2 = "D:\Retinopathy\RFMiD\AugmentedFlippedUD128"
pic = 0

# Get a list of image files in the directory
image_files = os.listdir(image_dir)

# Iterate over the image files and perform augmentation
for file_name in image_files:
    # Load the image
    image_path = os.path.join(image_dir, file_name)
    image = cv2.imread(image_path)
    pic = pic+1
    print(pic," Images Flipped")

    # Apply augmentation to the image
    augmented_images = seq(images=[image])

    # Save augmented images
    for i, augmented_image in enumerate(augmented_images):
        output_file_name = f"FUD_{file_name}"
        output_path = os.path.join(image_dir_2, output_file_name)
        cv2.imwrite(output_path, augmented_image)
