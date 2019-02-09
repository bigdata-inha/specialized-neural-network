from datasets.data_utils import load_cifar10
from clustering import make_cluster
import numpy as np
import glob
import datetime
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical


directory_list = glob.glob('cifar-10-batches/data_batch_*')
data, label = np.array([], dtype='uint8').reshape(0, 3072), np.array([])
for dir_batch in directory_list:
    partial_data, partial_label = load_data(dir_batch)
    data = np.vstack([data, partial_data])
    label = np.append(label, partial_label)

label = to_categorical(label)

data = data.astype('float32') / 255
label = label.astype('float32') / 255


val_data = data[0:10000]
val_label = label[0:10000]
partial_train_data = data[10000:]
partial_train_label = label[10000:]

print('training start:', datetime.datetime.now().strftime('%H:%M:%S'))
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(3072,)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(partial_train_data, partial_train_label, validation_data=(val_data, val_label), epochs=100,
          batch_size=1, verbose=0)
print('model fitting and save:', datetime.datetime.now().strftime('%H:%M:%S'))