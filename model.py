from set_gpu import get_session
import keras.backend.tensorflow_backend as KTF
res = KTF.set_session(get_session())
print "set_gpu result:", res
import os
import csv
import cv2
import numpy as np
import keras
import sklearn
import random

lines = []

data_dirs = ['track_01_ccw', 'track_01_cw', 'track_02_ccw', 'track_02_cw']
for _dir in data_dirs:
    driving_log_dir = os.path.join(os.getenv("HOME"), 'Downloads/linux_sim', _dir )
    driving_log_fn = os.path.join(driving_log_dir, 'driving_log.csv')
    if not os.path.isfile(driving_log_fn):
        continue
    print driving_log_fn
    with open(driving_log_fn) as fd:
        reader = csv.reader(fd)
        for line in reader:
            lines.append(line)

print "[----open file success!----]", driving_log_fn

def sample_generator(samples, batch_size=32):
    correction = 0.2
    num_samples = len(samples)
    while True:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for i, line in enumerate(batch_samples):
                corr_array = [0, correction, -correction]
                measurement = float(line[3])
                for j in range(3):
                    source_path = line[j]
                    _ = source_path.split('/')
                    current_path = os.path.join(os.getenv("HOME"), '/'.join(_[3:]))
                    image = cv2.imread(current_path, 1)
                    images.append(image)
                    images.append(np.fliplr(image))
                    measurements.append(measurement + corr_array[j])
                    measurements.append(-measurement - corr_array[j])
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

print "[-----read images finised-----]"

validation_split = 0.2
# compile and train the model using the generator function
train_samples = []
validation_samples = []
for line in lines:
    if random.random() <= validation_split:
        validation_samples.append(line)
    else:
        train_samples.append(line)

print len(train_samples), len(validation_samples)

train_generator = sample_generator(train_samples, batch_size=32)
validation_generator = sample_generator(validation_samples, batch_size=32)
test_gen = next(train_generator)
print test_gen[0].shape
print test_gen[1].shape

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

# simplest model
# model = Sequential()
# model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# model.add(Flatten())
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')
# print "[-----compile finished-----]"
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)
# print "[----- train finised-----]"

# Lenet
# model = Sequential()
# model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(10))
# model.add(Dense(1))

model = Sequential()
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Conv2D(filters=24, kernel_size=(5,5), subsample=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(filters=36, kernel_size=(5,5), subsample=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(filters=48, kernel_size=(5,5), subsample=(2, 2), padding='valid', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
print "[-----compile finished-----]"
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), epochs=3)
print "[----- train finised-----]"
model.save('model.h5')










