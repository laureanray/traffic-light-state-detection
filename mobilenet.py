import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

IMG_SHAPE = (1280, 720, 3)

cap = cv.VideoCapture("4.mp4")
ret, frame = cap.read()
resized_image = cv.resize(frame, None, fx=0.5, fy=0.5)

IMG_SHAPE = resized_image.shape
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

resized_image = np.expand_dims(resized_image, axis=0)
prediction = base_model.predict(resized_image)

print(prediction)

