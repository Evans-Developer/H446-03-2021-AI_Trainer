import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os
import cv2
import random
import pickle
import sys

MODEL_NAME = "XRAY_DETECT-{}".format(int(time.time()))

IMG_SIZE = 150

DATASETLOC = "C:\\Users\\joshc\\Documents\\Developing\\Repos\\H446-03-2021\\H446-03-2021-AI_Trainer\\DATASET"
CATEGORIES = ["Dangerous", "Non Dangerous"]

training_data = []

X = []
y = []


def generate_training_data():
    global X
    global y

    for category in CATEGORIES:
        path = os.path.join(DATASETLOC, category)
        class_id = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_binary = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_binary_resize = cv2.resize(img_binary, (IMG_SIZE, IMG_SIZE))
            training_data.append([img_binary_resize, class_id])

    random.shuffle(training_data)

    for feature, label in training_data:
        X.append(feature)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)

    save_training_data()


def save_training_data():
    global X
    global y

    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


def load_training_data():
    global X
    global y

    pickle_in = open("X.pickle", "rb")
    X = pickle.load(pickle_in)

    pickle_in = open("y.pickle", "rb")
    y = pickle.load(pickle_in)


if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "1":
        generate_training_data()
    elif mode == "2":
        load_training_data()

        X = X / 255.0

        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(X, y, batch_size=32, validation_split=0.1, epochs=1)

        model.save(MODEL_NAME)
    elif mode == "3":
        predict_file = sys.argv[2]
        model_name = sys.argv[3]

        print(predict_file)
        print(model_name)

        if predict_file is None:
            print("No file provided")
            quit()

        if model_name is None:
            print("No model name provided")

        img_array = cv2.imread(predict_file, cv2.IMREAD_GRAYSCALE)
        resize_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        resize_img_array = resize_img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        loaded_model = tf.keras.models.load_model(model_name)
        prediction = loaded_model.predict([resize_img_array])
        print(prediction)
        print(CATEGORIES[int(prediction[0][0])])