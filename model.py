from set_gpu import get_session
import keras.backend.tensorflow_backend as KTF
res = KTF.set_session(get_session())
print "set_gpu result:", res
import os
import csv
import cv2
import numpy as np
import keras

lines = []
driving_log_dir = os.path.join(os.getenv("HOME"), 'Downloads/linux_sim/record_data' )
driving_log_fn = os.path.join(driving_log_dir, 'driving_log.csv')

with open(driving_log_fn) as fd:
    reader = csv.reader(fd)
    for line in reader:
        lines.append(line)

print "[----open file success!----]", driving_log_fn

images = []
measurements = []
correction = 0.2
for i, line in enumerate(lines):
    for j in range(3):
        source_path = line[j]
        _, fn = os.path.split(source_path)
        current_path = os.path.join(driving_log_dir, 'IMG', fn)
        image = cv2.imread(current_path, 1)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement + (j - 1) * correction)
    if i % 200 == 0:
        print "[read images] %5d/%5d " % (i, len(lines))

print "[-----read images finised-----]"

X_train = np.array(images)
y_train = np.array(measurements)
print "[----- transform to numpy array ok-----]"


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D

# simplest model
# model = Sequential()
# model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# model.add(Flatten())
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')
# print "[-----compile finished-----]"
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)
# print "[----- train finised-----]"

model = Sequential()
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
print "[-----compile finished-----]"
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)
print "[----- train finised-----]"
model.save('model.h5')










