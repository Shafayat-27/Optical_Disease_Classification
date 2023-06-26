import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from natsort import natsorted

# Set up the image augmentation sequence
# seq = iaa.Sequential([
#     iaa.ColorJitter(brightness=1.5)  # Increase brightness by a factor of 1.5
# ])

# seq = iaa.Sequential([
#     iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect color
#     # iaa.LinearContrast((0.75, 1.5)), # improve or worsen the contrast
#     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5) # add gaussian noise to images
# ])

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=False)

# Specify the path to the directory containing the images
image_dir = "D:\Retinopathy\RFMiD\Train_Set\demo"
image_dir_2 = "D:\Retinopathy\RFMiD\Train_Set\demo2"

# Get a list of image files in the directory
image_files = os.listdir(image_dir)
image_files = natsorted(image_files)
print(image_files)

last_file_value = 10

# Iterate over the image files and perform augmentation
for file_name in image_files:
    # Load the image
    image_path = os.path.join(image_dir, file_name)
    # print(image_path)
    image = cv2.imread(image_path)

    # Apply augmentation to the image
    augmented_images = seq(images=[image])

    # Save augmented images
    for i, augmented_image in enumerate(augmented_images):
        output_file_name = f"{last_file_value+i+1}.png"
        # print(output_file_name)
        output_path = os.path.join(image_dir, output_file_name)
        # cv2.imwrite(output_path, augmented_image)

    last_file_value = last_file_value + len(augmented_images)