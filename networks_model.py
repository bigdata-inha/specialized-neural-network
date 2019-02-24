from keras import models
from keras import layers


def full_model(num_classes, input_shape):
    model = models.Sequential()

    # Block1
    model.add(
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block4
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))

    # FC layers
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(4096, activation='relu', name='fc1'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2048, activation='relu', name='fc2'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def layer_reduced_model(num_classes, input_shape):
    model = models.Sequential()

    # Block1
    model.add(
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block4
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))


    # FC layers
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(4096, activation='relu', name='fc1'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2048, activation='relu', name='fc2'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def filter_reduced_model(num_classes, input_shape):
    model = models.Sequential()

    # Block1
    model.add(
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block4
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3'))

    # FC layers
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(2048, activation='relu', name='fc1'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu', name='fc2'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def both_reduced_model(num_classes, input_shape):
    model = models.Sequential()

    # Block1
    model.add(
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block4
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2'))

    # FC layers
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(2048, activation='relu', name='fc1'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu', name='fc2'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model



