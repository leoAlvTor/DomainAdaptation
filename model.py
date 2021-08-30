import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


def read_data():
    return pd.read_csv('/media/edutech-pc06/Elements/DataSet/ClasificacionPorContenido/dataframe.csv')
    # train, test = train_test_split(dataframe, test_size=0.2, random_state=42)
    # print(f'Train shape: {train.shape}, Test shape: {test.shape}')
    # return train, test


dataframe = read_data()
IMG_SIZE = (256, 256)

image_data_generator = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_generator = image_data_generator.flow_from_dataframe(
    dataframe=dataframe,
    x_col='path',
    y_col='class',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32
)

test_generator = image_data_generator.flow_from_dataframe(
    dataframe=dataframe,
    x_col='path',
    y_col='class',
    target_size=IMG_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    subset='validation'
)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

history = model.fit_generator(
    train_generator,
    epochs=5,
    validation_data=test_generator,
    steps_per_epoch=128,
    validation_steps=64)
