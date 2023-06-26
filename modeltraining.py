import numpy as np
import pandas as pd
np.random.seed(1000)

import matplotlib.pyplot as plt
import os
import cv2
import keras
from keras.optimizers import SGD
from keras.utils import load_img, img_to_array
from tqdm import tqdm
from sklearn.model_selection import train_test_split

os.environ['KERAS_BACKEND'] = 'tensorflow' 

image_directory = 'D:\Retinopathy\RFMiD\DatasetTwo\All_Original_Plus_Augmented/'
df = pd.read_csv('D:\Retinopathy\RFMiD\Train_Set/RFMiD_Training_Labels_7680.csv') 
# print(df.columns)
SIZE = 128
X_dataset = []

for i in tqdm(range(df.shape[0])):
    img = load_img(image_directory +str(df['ID'][i])+'.png', target_size=(SIZE,SIZE,3))
    img = img_to_array(img)
    img = img/255.
    X_dataset.append(img)
    
X = np.array(X_dataset)
# print(X.shape)
y = np.array(df.drop(['ID', 'Disease_Risk'], axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.1)

        
INPUT_SHAPE = (SIZE, SIZE, 3) 
inp = keras.layers.Input(shape=INPUT_SHAPE)

conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
                               activation='relu', padding='same')(inp)
conv2 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
                               activation='relu', padding='same')(conv1)
conv3 = keras.layers.Conv2D(32, kernel_size=(3, 3), 
                               activation='relu', padding='same')(conv2)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
# norm1 = keras.layers.BatchNormalization(axis = -1)(pool1)
drop1 = keras.layers.Dropout(rate=0.25)(pool1)

conv4 = keras.layers.Conv2D(64, kernel_size=(3, 3), 
                               activation='relu', padding='same')(drop1)
conv5 = keras.layers.Conv2D(64, kernel_size=(3, 3), 
                               activation='relu', padding='same')(conv4)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)
# norm2 = keras.layers.BatchNormalization(axis = -1)(pool2)
drop2 = keras.layers.Dropout(rate=0.25)(pool2)

flat = keras.layers.Flatten()(drop2)

hidden1 = keras.layers.Dense(512, activation='relu')(flat)
# norm3 = keras.layers.BatchNormalization(axis = -1)(hidden1)
drop3 = keras.layers.Dropout(rate=0.5)(hidden1)

out = keras.layers.Dense(45, activation='sigmoid')(drop3)


model = keras.Model(inputs=inp, outputs=out)

optimizer = SGD(learning_rate=0.01, momentum=0.9, decay=1e-6)
model.compile(  optimizer=optimizer,
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
print(model.summary())


history = model.fit(    X_train, 
                        y_train, 
                        batch_size = 32, 
                        verbose = 1, 
                        epochs = 1,  
                        validation_data=(X_test, y_test),
                        validation_split= 0.1,
                        shuffle = False
                     )

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# plt.plot(epochs, acc, 'y', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()