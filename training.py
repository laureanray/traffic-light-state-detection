import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

# Data Preprocessing
base_dir = "./dataset"

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat pictures
train_amber_dir = os.path.join(train_dir, 'amber')
print ('Total training amber images:', len(os.listdir(train_amber_dir)))

# Directory with our training dog pictures
train_green_dir = os.path.join(train_dir, 'green')
print ('Total training green images:', len(os.listdir(train_green_dir)))

# Directory with our training dog pictures
train_red_dir = os.path.join(train_dir, 'red')
print ('Total training green images:', len(os.listdir(train_red_dir)))

# Directory with our training dog pictures
train_zero_dir = os.path.join(train_dir, 'zero')
print ('Total training zero images:', len(os.listdir(train_zero_dir)))



# Directory with our training cat pictures
validation_amber_dir = os.path.join(validation_dir, 'amber')
print ('Total validation amber images:', len(os.listdir(validation_amber_dir)))

# Directory with our training dog pictures
validation_green_dir = os.path.join(validation_dir, 'green')
print ('Total validation green images:', len(os.listdir(validation_green_dir)))

# Directory with our training dog pictures
validation_red_dir = os.path.join(validation_dir, 'red')
print ('Total validation red images:', len(os.listdir(validation_red_dir)))

# Directory with our training dog pictures
validation_zero_dir = os.path.join(validation_dir, 'zero')
print ('Total validation zero images:', len(os.listdir(validation_zero_dir)))

image_size = 160
batch_size = 16

# Rescale all images by 1./255 and apply image augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                train_dir,  # Source directory for the training images
                target_size=(image_size, image_size),
                batch_size=batch_size,
                # Since we use binary_crossentropy loss, we need binary labels
                class_mode='categorical')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
                validation_dir, # Source directory for the validation images
                target_size=(image_size, image_size),
                batch_size=batch_size,
                class_mode='categorical')

IMG_SHAPE = (image_size, image_size, 3)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3,), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3,), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train
epochs = 20
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

history = model.fit_generator(train_generator,
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator,
                              validation_steps=validation_steps)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()

model_json = model.to_json()
with open("mode.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")

print("Saved model to disk")


