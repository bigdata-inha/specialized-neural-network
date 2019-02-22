from data_pipe import load_tiny_imagenet
import custom_callback
import numpy as np
import time
import datetime
from keras import backend
from keras import models
from keras import layers
from keras import optimizers
from keras import initializers
from keras import regularizers
from keras import callbacks
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# data load
train_data, train_labels, test_data, test_labels = load_tiny_imagenet("datasets/tiny-imagenet-200")
train_data, test_data = (train_data.astype('float32') - 128.0) / 128.0, (test_data.astype('float32') - 128.0) / 128.0
train_labels, test_labels = to_categorical(train_labels, 200), to_categorical(test_labels, 200)

# sample data
idx = np.random.choice(len(train_data), 10000)
train_data, train_labels = train_data[idx], train_labels[idx]
idx = np.random.choice(len(test_data), 1000)
test_data, test_labels = test_data[idx], test_labels[idx]

# make model
