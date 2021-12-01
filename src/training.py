import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEIVCES"] = "5"

import argparse
import pandas as pd
import numpy as np
import mylib.data_preprocessing as dpp
import mylib.data_augmentation as daug

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers import LeakyReLU
import matplotlib.pyplot as plt
import math

#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#    except RuntimeError as e:
#        print(e)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)
if __name__ == "__main__":
    # Read dataset from command line
    key_word = "--dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument(key_word, required=False, default='../data/skeleton_raw.csv')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=50)
    input = parser.parse_args().dataset
    init_lr = parser.parse_args().lr
    epoch = parser.parse_args().epoch

    # Loading training data
    try:
        raw_data = pd.read_csv(input, header=0)
    except:
        print("Dataset not exists.")
    # X: input, Y: output
    dataset = daug.data_augmentation(raw_data)
    X = dataset[:, 0:36].astype(float)
    Y = dataset[:, 36]

    # Data pre-processing
    # X = dpp.head_reference(X)
    X_pp = []
    for i in range(len(X)):
        X_pp.append(dpp.pose_normalization(X[i]))
    X_pp = np.array(X_pp)
    print('Xpp shape: ', X_pp.shape)
    # Encoder the class label to number
    # Converts a class vector (integers) to binary class matrix
    encoder = LabelEncoder()
    encoder_Y = encoder.fit_transform(Y)
    matrix_Y = np_utils.to_categorical(encoder_Y)
    print(Y[0], ": ", encoder_Y[0])
    print(Y[17499], ": ", encoder_Y[17499])
    print(Y[33383], ": ", encoder_Y[33383])
    print(Y[56365], ": ", encoder_Y[56365])
    print(Y[80206], ": ", encoder_Y[80206])

    # Split into training and testing data
    # random_state:
    X_train, X_test, Y_train, Y_test = train_test_split(X_pp, matrix_Y, test_size=0.1, random_state=42)

    # Build DNN model with keras
    model = Sequential()
    model.add(Dense(units=512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=5, activation='softmax'))

    # lr scheduler
    def scheduler(epoch, lr):
        drop_rate = 0.1
        epochs_drop = 20.0
        return init_lr * math.pow(drop_rate, math.floor(epoch/epochs_drop))

    # Training
    # optimiser: Adam with learning rate 0.0001
    # loss: categorical_crossentropy for the matrix form matrix_Y
    # metrics: accuracy is evaluated for the model
    model.compile(optimizer=Adam(init_lr), loss='categorical_crossentropy', metrics=['accuracy'])
    # batch_size: number of samples per gradient update
    # epochs: how many times to pass through the whole training set
    # verbose: show one line for every completed epoch
    #callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    history = model.fit(X_train, Y_train, batch_size=32, epochs=epoch, verbose=2, validation_data=(X_test, Y_test))
    
    # plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Test'], loc='upper left')
    plt.savefig(f'result_custom/model_accuracy_lr_{init_lr}.png')
    plt.clf()

    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f'result_custom/loss_lr_{init_lr}.png')

    model.summary()

    # Save the trained model
    model.save('./model/action_recognition.h5')
