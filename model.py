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
driving_log_dir = '/home/messi/Downloads/linux_sim/record_data' 
driving_log_fn = os.path.join(driving_log_dir, 'driving_log.csv')

with open(driving_log_fn) as fd:
    reader = csv.reader(fd)
    for line in reader:
        lines.append(line)

print "[----open file success!----]", driving_log_fn

images = np.zeros((len(lines), 160, 320, 3), dtype=np.uint8)
measurements = np.zeros(len(lines)) 
for i, line in enumerate(lines):
    source_path = line[0]
    _, fn = os.path.split(source_path)
    current_path = os.path.join(driving_log_dir, 'IMG', fn)
    image = cv2.imread(current_path, 1)
    images[i] = image
    measurement = float(line[3])
    measurements[i] = measurement
    if i % 1000 == 0:
        print "[read images] %5d/%5d " % (i, len(lines))

print "[-----read images finised-----]"

X_train = images
y_train = measurements
print "[----- transform to numpy array ok-----]"


from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
print "[-----compile finished-----]"
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=30)
print "[----- train finised-----]"
model.save('model.h5')










