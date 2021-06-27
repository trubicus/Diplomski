import numpy as np
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import tensorflow.keras.optimizers

tf.autograph.set_verbosity(0)

max_len = 0
final_len = 5300

x_train = np.load("x_train_normal.npy")
y_train = np.load("y_train_normal.npy")
x_test = np.load("x_test_normal.npy")
y_test = np.load("y_test_normal.npy")

# Gradnja modela neuronske mreže
model = Sequential()
model.add(LSTM(128, input_shape=(12, final_len), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(6, activation='softmax'))

#opt = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
#model.fit(x_train, y_train, epochs=60, validation_data=(x_test, y_test), batch_size=30, shuffle=True)
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=200, shuffle=True)
model.save("benchmark")