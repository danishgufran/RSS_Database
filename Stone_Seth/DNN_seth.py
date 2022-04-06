import copy
from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D , LSTM
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import*
import random as random
import time
import matplotlib.pyplot as plt
import Seth
from Seth import fetch_seth, Devices, Floorplan, get_mac_ids

def train_data():
    # dfs is a list of dataframes
# meta is a dataframe with meta data

#getting train data
    train_fp, train_meta = fetch_seth(
    Devices.lg,
    Floorplan.BASEMENT,
    ci = 0,
    base_path="temp/clean/"  # <-- this would be 'seth/temp/clean' from outside this dir
)
    train_fp = train_fp.sample(frac=1).reset_index(drop=True)
    train_aps = get_mac_ids(train_fp.columns)
    train_x = train_fp[train_aps].values
    train_x = (train_x + 100)/100
    train_y = (train_fp["label"]).values
    return train_x, train_y, train_aps

def test_data(itr, train_aps):
    #getting test data
    test_fp, test_meta = fetch_seth(
    Devices.name,
    Floorplan.BASEMENT,
    ci = itr,
    base_path="temp/clean/"  # <-- this would be 'seth/temp/clean' from outside this dir
)
    test_aps = get_mac_ids(test_fp.columns)
    missing_aps = list(set(train_aps)-set(test_aps))
    test_fp[missing_aps] = -100
    test_x = test_fp[train_aps].values
    test_x = (test_x + 100)/100
    test_y = (test_fp["label"]).values
    return test_x, test_y

def mean_cal(group):
    for i in range (0, len(group)):
        if group[i] < 0:
            group[i] = -group [i]
    mean_diff = np.mean(group)
    return mean_diff

def main():

    name = ["lg","blu"]
    train_x, train_y, train_aps = train_data()
    num_classes = 48
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=100)]

    error = []
    mean_error = []

    model = Sequential()
    model.add(Flatten(input_dim=203))
    model.add(tf.keras.layers.GaussianNoise(0.10))
    model.add(Dropout(0.2))
    # model.add(Dense(32, input_dim=203, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    opt = Adam(learning_rate=0.001)

    model.compile(loss=sparse_categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

    model.fit(train_x, train_y,
            epochs=1000,
            callbacks = callbacks,
            validation_split =0.2 ,
            verbose=1)

    for z in range(0, len(name)):
        test_x, test_y = test_data(0, name[0], train_aps)
        pred = np.argmax(model.predict(test_x), axis=1)
        acc = sum([(test_y[i])==(pred[i]) for i in range(228)])/228
        print('Test accuracy:', acc)


        for i in range (0, len(pred)):
            error.append(test_y[i]-pred[i])
            
        mean_error.append(mean_cal(error))

        print("Mean Error  = ", mean_error)



if __name__ == '__main__':
    main()


