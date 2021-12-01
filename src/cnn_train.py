import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
from numpy import array
from numpy import hstack
import pandas as pd
import mylib.data_preprocessing as dpp

import tensorflow as tf
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.optimizers import Adam, SGD

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":
    
    # split a multivariate sequence into samples
    def split_sequences(X_data, Y_data, n_steps):
        X, y = list(), list()
        for i in range(len(X_data)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(X_data):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = X_data[i:end_ix, :], Y_data[end_ix-1]
            X.append(seq_x)                                                                      
            y.append(seq_y)
        return array(X), array(y)

    raw_data = pd.read_csv('./data/skeleton_raw_custom.csv', header=0, index_col=0)
    dataset = raw_data.values
    X = dataset[:, 0:36].astype(float)
    Y = dataset[:, 36]

    # data preprocessing, change bbox criteria
    X_pp = []
    for i in range(len(X)):
        X_pp.append(dpp.pose_normalization(X[i]))
    X_pp = np.array(X_pp)
    

    # a number of time steps
    n_steps = 5
    # convert into sequential data
    seq_X, seq_Y = split_sequences(X_pp, Y, n_steps)
    # encoder the class label to number
    # converts a class vector to binary class matrix
    encoder = LabelEncoder()
    encoder_Y = encoder.fit_transform(seq_Y)
    matrix_Y = np_utils.to_categorical(encoder_Y)

    X_train, X_valid, y_train, y_valid = train_test_split(seq_X, matrix_Y, test_size=0.1, shuffle=True)
    # number of features
    n_features = 26

    # model definition
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, verbose=2, validation_data=(X_valid, y_valid))    
