import numpy as np
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf

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

#print("setting up training data...")
x_train, y_train = load_data(db_dir)
print(len(x_train), len(y_train))
x_train = match_len(x_train)
print(x_train[0].shape)

#print("setting up test data...")
x_test, y_test = load_data(test_dir)
print(len(x_test), len(y_test))
x_test = match_len(x_test)


#print(x_train[0].shape)
y_train = np.array(y_train, dtype=np.float32)
x_train = np.array(x_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

np.save('x_train.npy', x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)
print(os.path.abspath("."))
print("saved data")