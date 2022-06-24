#!/usr/bin/env python
# -*- coding=utf-8 -*-



import tensorflow as tf
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import os.path as path
import os
import cv2

IMAGEPATH = 'image'
dirs = os.listdir(IMAGEPATH)
X=[]
Y=[]
print(dirs)
w=32 # 224
h=32 # 224
i=0
for name in dirs:
    file_paths = glob.glob(path.join(IMAGEPATH+"/"+name, '*.*'))
    for path3 in file_paths:
        img = cv2.imread(path3)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X.append(im_rgb)
        Y.append(i)
    i=i+1


X = np.asarray(X)
Y = np.asarray(Y)

X = X.astype('float32')
X=X/255
X=X.reshape(X.shape[0],w,h,3)

category=len(dirs)
dim=X.shape[1]
x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.05)
# 將數字轉為 One-hot 向量
y_train2 = tf.keras.utils.to_categorical(y_train, category)
y_test2 = tf.keras.utils.to_categorical(y_test, category)



# 載入資料（將資料打散，放入 train 與 test 資料集）

print(x_train.shape)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                            rotation_range=25 ,
                            width_shift_range=[-3,3],
                            height_shift_range=[-3,3] ,
                            zoom_range=0.3 ,
							data_format='channels_last')


# 建立模型
model = tf.keras.models.Sequential()
# 加入 2D 的 Convolution Layer，接著一層 ReLU 的 Activation 函數
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                 padding="same",
                 activation='relu',
                 input_shape=(w,h,3)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))


learning_rate = 0.001
opt1 = tf.keras.optimizers.Adam(lr=learning_rate)
model.compile(
    optimizer=opt1,
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy'])

model.summary()
with open("model_ImageDataGenerator_myImage.json", "w") as json_file:
    json_file.write(model.to_json())


try:
    with open('model_ImageDataGenerator_myImage.h5', 'r') as load_weights:
        model.load_weights("model_ImageDataGenerator_myImage.h5")
except IOError:
    print("File not exists")


checkpoint = tf.keras.callbacks.ModelCheckpoint("model_ImageDataGenerator_myImage.h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', save_freq=1)
# 訓練模型

trainData=datagen.flow(x_train,y_train2,batch_size=64)
history = model.fit_generator(trainData,
							  epochs=1000,
							  callbacks=[checkpoint]
                              )



#測試
score = model.evaluate(x_test, y_test2, batch_size=128)
# 輸出結果
print("score:",score)

predict = model.predict(x_test)
print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]),np.argmax(predict[2]))

predict2 = model.predict_classes(x_test)
print("predict_classes:",predict2)
print("y_test",y_test[:])
for t1 in predict2:
    print(dirs[t1])

img=x_test[0]
img=img.reshape(w,h,3)
img=img*255
img = img.astype('uint8')
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
im_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
i = np.argmax(predict[0])
str1 = dirs[i] + "   " + str(predict[0][i])
print(str1)
im_bgr = cv2.putText(im_bgr, str1, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0, 0), 1, cv2.LINE_AA)
cv2.imshow('image', im_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()