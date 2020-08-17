import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import Adam
import cv2
import numpy as np

def init_cnn_model():
    model=Sequential()

    model.add(Conv2D(64,(3,3),activation='elu',input_shape=(32,32,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),activation='elu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,(3,3),activation='elu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),activation='elu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(256,(3,3),activation='elu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(3,3),activation='elu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(3,3),activation='elu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(512,(3,3),activation='elu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512,(3,3),activation='elu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512,(3,3),activation='elu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512,(3,3),activation='elu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(512,activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(512,activation='elu'))
    model.add(BatchNormalization())
    model.add(Dense(10,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=3e-4), metrics=['accuracy'])

    return model

img = input("Enter image path: ")

if img:
    image = cv2.imread(img, cv2.IMREAD_COLOR)

    image = cv2.resize(image, (32, 32))

    image = image/255.0

    image=image.reshape((1,32,32,3))

    model = init_cnn_model()

    model.load_weights('cifar_model_weights.h5')
    #model.load_weights('cifar_model.h5')

    prediction = model.predict(image)

    print("Class ID: {}".format(np.argmax(prediction)))

    exit()

else:
    print("Invalid Image path")

    exit()

