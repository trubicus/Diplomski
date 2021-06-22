import numpy as np
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import tensorflow.keras.optimizers

#tf.compat.v1.disable_eager_execution()
tf.autograph.set_verbosity(0)


# Odabir stupaca - IMU senzori sa desne nadkoljenice i potkoljenice
# Stupci   |rsacc|  |rsgyro|    |-rtacc-|  |-rtgyro-|
columns = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)

actions = ["bicycling", "running", "sitting_in_car", "sitting", "standing", "walking"]

max_len = 0
final_len = 21155

def load_file(path_to_file):
    return np.genfromtxt(path_to_file, delimiter='\t', skip_header=4, usecols=columns, dtype=np.float32)

def load_data(path):
    x = []
    y = []
    data = os.listdir(path)
    random.shuffle(data)
    for sample in data:
        x.append(np.transpose(load_file(path+sample)))
        for action in actions:
            if action in sample:
                out = [0.0 for i in actions]
                out[actions.index(action)] = 1.0
                y.append(np.transpose(np.array(out, dtype=np.float32)))
                break

    print("returning data")
    #return np.array(x, dtype=np.ndarray), np.array(y, dtype=np.ndarray)
    return x, y

def find_max_len(data_arr):
    global max_len
    for i in data_arr:
        i_len = i.shape[1]
        if i_len > max_len:
            max_len = i_len

def match_len(data_arr):
    global max_len
    if max_len == 0:
        find_max_len(data_arr)

    for i in range(len(data_arr)):
        data_arr[i] = np.pad(data_arr[i], ((0, 0), (0, max_len - data_arr[i].shape[1])), mode='constant')
        
    return data_arr


db_dir = "/home/truba/Documents/Faks/Diplomski/db/HuGaDB/Data/"
test_dir = "/home/truba/Documents/Faks/Diplomski/db/HuGaDB/Test/"

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")



# Gradnja modela neuronske mreže
model = Sequential()
model.add(LSTM(128, input_shape=(12, final_len), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(6, activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

#model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), batch_size=3)
