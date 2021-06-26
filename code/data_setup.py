import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Odabir stupaca - IMU senzori sa desne nadkoljenice i potkoljenice
# Stupci   |rsacc|  |rsgyro|    |-rtacc-|  |-rtgyro-|
columns = (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)

actions = ["bicycling", "running", "sitting_in_car", "sitting", "standing", "walking"]

max_len = 21200
final_len = max_len
g = 9.81

def load_file(path_to_file):
    return np.genfromtxt(path_to_file, delimiter='\t', skip_header=4, usecols=columns, dtype=np.float32)

def normalize(data):
    normal = np.array([
        data[0] / (2*g*1000),
        data[1] / (2*g*1000),
        data[2] / (2*g*1000),
        data[3] / 2000000,
        data[4] / 2000000,
        data[5] / 2000000,
        data[6] / (2*g*1000),
        data[7] / (2*g*1000),
        data[8] / (2*g*1000),
        data[9] / 2000000,
        data[10] / 2000000,
        data[11] / 2000000,
    ])
    return normal


def load_data(path):
    x = []
    y = []
    data = os.listdir(path)
    random.shuffle(data)
    for sample in data:
        for split in np.array_split(pad_data(np.transpose(normalize(load_file(path+sample)))), 4, axis=1):
            x.append(split)
        for action in actions:
            if action in sample:
                out = [0.0 for i in actions]
                out[actions.index(action)] = 1.0
                for j in range(5):
                    y.append(np.transpose(np.array(out, dtype=np.float32)))
                break

    return x, y


def pad_data(data):
    return  np.pad(data, ((0, 0), (0, max_len - data.shape[1])), mode='wrap')

db_dir = "/home/truba/Documents/Faks/Diplomski/db/HuGaDB/Data/"
test_dir = "/home/truba/Documents/Faks/Diplomski/db/HuGaDB/Test/"

print("setting up training data...")
x_train, y_train = load_data(db_dir)
print(len(x_train), len(y_train))
print(x_train[0].shape, y_train[0].shape)

print("setting up test data...")
x_test, y_test = load_data(test_dir)
print(len(x_test), len(y_test))


y_train = np.array(y_train, dtype=np.float32)
x_train = np.array(x_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)
print(x_train.shape, y_train.shape)

np.save('x_train_normal.npy', x_train)
np.save("y_train_normal.npy", y_train)
np.save("x_test_normal.npy", x_test)
np.save("y_test_normal.npy", y_test)
print(os.path.abspath("."))
print("saved data")