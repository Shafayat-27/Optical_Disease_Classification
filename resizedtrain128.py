import cv2
import numpy as np
import os

path = 'D:\Retinopathy\RFMiD\Train_Set\Train_Set'
path_for_resized_image = 'D:\Retinopathy\RFMiD\ResizedTrain224'

for filename in os.listdir(path):
    img = cv2.imread(os.path.join(path,filename))
    print('Image No.', filename)
    
    resized_image = cv2.resize(img, (224, 224))

    cv2.imwrite(os.path.join(path_for_resized_image , filename), resized_image)

