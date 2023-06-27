import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from natsort import natsorted

seq = iaa.Sequential([
    iaa.Fliplr(p=1.0),
    iaa.Flipud(p=1.0)
])

image_dir = "D:\Retinopathy\RFMiD\DatasetThree\Aug_1920"
image_dir_2 = "D:\Retinopathy\RFMiD\DatasetThree\Flipped_Aug_1920_UD_LR"

image_files = os.listdir(image_dir)
image_files = natsorted(image_files)

last_file_value = 3840

for file_name in image_files:
    image_path = os.path.join(image_dir, file_name)
    image = cv2.imread(image_path)
    

    augmented_images = seq(images=[image])

    for i, augmented_image in enumerate(augmented_images):
        output_file_name = f"{last_file_value+i+1}.png"
        print(output_file_name)
        output_path = os.path.join(image_dir_2, output_file_name)
        cv2.imwrite(output_path, augmented_image)
        
    
    last_file_value = last_file_value + len(augmented_images)
    
