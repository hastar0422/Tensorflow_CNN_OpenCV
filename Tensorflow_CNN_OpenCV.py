"""
作品名稱     : Tensorflow_OpenCV
__author__ : Ning甯詠城
"""

#!/usr/bin/env python
# -*- coding=utf-8 -*-



import tensorflow as tf
import cv2
import numpy as np
import os.path as path
import os

IMAGEPATH = 'image'
dirs = os.listdir(IMAGEPATH)



json_file = open('model_ImageDataGenerator_myImage.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights("model_ImageDataGenerator_myImage.h5")
model.compile(loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adadelta(),
            metrics=['accuracy'])

model.summary()
cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    if ret==True:
        resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = image.astype('float32')                                                                 #標準化
        image = image / 255                                                                             #標準化
        predict = model.predict(image)
        i=np.argmax(predict[0])
        str1 =dirs[i] +"   "+str(predict[0][i])
        print(str1)
        img = cv2.putText(img, str1, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2,cv2.LINE_AA)
        cv2.imshow('image',img)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




