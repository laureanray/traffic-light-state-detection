# from keras.models import Sequential
# from keras.layers import Dense
# from keras.models import model_from_json
# from keras.utils import to_categorical

import cv2 as cv
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
json_file = open("mode.json", "r")
loaded_model_json = json_file.read()

json_file.close()

loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


cap = cv.VideoCapture("4.mp4")


state, initial_frame = cap.read()

resized_frame = cv.resize(initial_frame, None, fx=0.5, fy=0.5)
x, y, w, h = cv.selectROI(resized_frame)


cv.waitKey(0)
cv.destroyAllWindows()


while(True):
    ret, frame = cap.read()
    res_frame = cv.resize(frame, None, fx=0.5, fy=0.5)
    roi = res_frame[y:y+h, x:x+w]

    img = cv.resize(roi, (160, 160), interpolation=cv.INTER_LINEAR)
    cv.imshow('roi', roi)
    # cv.imshow('resized', res_frame)
    img = np.expand_dims(img, axis=0)
    prediction = loaded_model.predict(img)
    result = np.where(prediction == np.amax(prediction))
    if result[1] == 0:
        print("Amber " + str(np.amax(prediction) * 100) + "%")
    elif result[1] == 1:
        print("Green " + str(np.amax(prediction) * 100)  + "%")
    elif result[1] == 2:
        print("Red " + str(np.amax(prediction) * 100)  + "%")
    else:
        print("Zero " + str(np.amax(prediction) * 100)  + "%")



    if cv.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv.destroyAllWindows()
