from os import listdir
from os import getcwd
from os.path import isfile, splitext
import numpy as np
import keras
import cv2
from matplotlib.pyplot import imread
import os
from sklearn.model_selection import train_test_split


def read_data_from_files(
    *fpaths,
):  # read data from all files in the listed directories *fpaths and process the images into (np.array, str) tuples
    cwd = getcwd() + "\\"
    return (
        (imread(cwd + "{}\\{}".format(dir, fil)), dir)
        for dir in fpaths
        for fil in listdir(cwd + dir)
    )


def preprocess_data(datagenerator, xscale, yscale, grey=True):
    if grey:
        return (
            (inetrpolate(greyscale(feature), xscale, yscale), label)
            for (feature, label) in datagenerator
            if feature.ndim == 3
        )
    else:
        return (
            (inetrpolate(feature, xscale, yscale), label)
            for (feature, label) in datagenerator
            if feature.ndim == 3
        )


def inetrpolate(arr, *scale):
    return cv2.resize(arr, dsize=scale, interpolation=cv2.INTER_CUBIC)


def greyscale(img):
    return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114


beefile = "Bees"
notbeefile = "Not_Bees"

data = read_data_from_files(beefile, notbeefile)
processed_data = preprocess_data(data, 100, 100, False)

features, labels = zip(*list(processed_data))

features = np.array(features, dtype=np.float32)
labels = np.array([0 if i == notbeefile else 1 for i in labels])

train_x, test_x, train_y, test_y = train_test_split(features, labels, train_size=0.85)

import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (3, 3), padding="same", activation=tf.nn.relu, input_shape=(100, 100, 1)
        ),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ]
)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(
    train_x.reshape(len(train_x), 100, 100, 1),
    train_y,
    epochs=5,
    validation_split=0.176,
)  # 0.176 = 0.15/0.85

model.evaluate(test_x, test_y)
