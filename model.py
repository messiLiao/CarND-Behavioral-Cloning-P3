import os
import csv
import cv2
import numpy as np
import keras

lines = []
driving_log_dir = '/home/figo/Downloads/linux_sim/record_data' 
driving_log_fn = os.path.join(driving_log_dir, 'driving_log.csv')

with open(driving_log_fn) as fd:
    reader = csv.reader(fd)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    _, fn = os.path.split(source_path)
    current_path = os.path.join(driving_log_dir, 'IMG', fn)
    image = cv2.imread(current_path, 1)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurements)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compole(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')










