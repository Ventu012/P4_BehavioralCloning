import csv
import cv2
import numpy as np
import os
import numpy as np
import keras
import socketio

from PIL import Image
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K

lines = []
car_images = []
steering_angles = []

with open('data/driving_log.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        path = "data/IMG/" # fill in the path to your training IMG directory
        img_center = np.asarray(Image.open(path + row[0].split('\\')[-1]))
        img_left = np.asarray(Image.open(path + row[1].split('\\')[-1]))
        img_right = np.asarray(Image.open(path + row[2].split('\\')[-1]))

        # add images and angles to data set
        car_images.extend([img_center, img_left, img_right])
        steering_angles.extend([steering_center, steering_left, steering_right])
        
        # augmenting data: flipping images and steering
        img_center_flipped = np.fliplr(img_center)
        img_left_flipped = np.fliplr(img_left)
        img_right_flipped = np.fliplr(img_right)
        car_images.extend([img_center_flipped, img_left_flipped, img_right_flipped])
        steering_center_flipped = -steering_center
        steering_left_flipped = -steering_left
        steering_right_flipped = -steering_right
        steering_angles.extend([steering_center_flipped, steering_left_flipped, steering_right_flipped])

X_train = np.array(car_images)
y_train = np.array(steering_angles)


'''
The CNN architecture used is the one used in the NVIDIA's End to End Learning for Self-Driving Cars paper.
Reference: https://arxiv.org/pdf/1604.07316v1.pdf
'''
# Defining Keras Sequential Model
model = Sequential()

# Performing image cropping to get rid of the irrelevant parts of the image (the sky and the hood)
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# Pre-Processing the image applying normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# CNN Layers
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5, 5),strides=(2, 2), activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3) ,activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3),activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1))
print(model.summary())

# Training the model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('model/model.h5')
print('The model.h5 file has been created!') 