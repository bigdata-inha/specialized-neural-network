import pickle
import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def load_cifar10_batch(directory):
    """ load batch of cifar """
    with open(directory, 'rb') as fo:
        datadict = pickle.load(fo, encoding='bytes')
    X = np.array(datadict[b'data'])
    Y = np.array(datadict[b'labels'])
    return X, Y

def load_cifar10(directory):
    """ load all of cifar """
    train_data = []
    train_labels = []
    for b in range(1, 6):
        f = os.path.join(directory, 'data_batch_%d' % (b,))
        X, Y = load_cifar10_batch(f)
        train_data.append(X)
        train_labels.append(Y)
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    del X, Y
    test_data, test_labels = load_cifar10_batch(os.path.join(directory, 'test_batch'))
    return train_data, train_labels, test_data, test_labels

def load_cifar10_img_form(directory):
    """ load all of cifar as image form """
    train_data, train_labels, test_data, test_labels = load_cifar10(directory)
    R, testR = train_data[:, :1024].reshape(-1, 32, 32, 1), test_data[:, :1024].reshape(-1, 32, 32, 1)
    G, testG = train_data[:, 1024:2048].reshape(-1, 32, 32, 1), test_data[:, 1024:2048].reshape(-1, 32, 32, 1)
    B, testB = train_data[:, 2048:].reshape(-1, 32, 32, 1), test_data[:, 2048:].reshape(-1, 32, 32, 1)
    train_data, test_data = np.concatenate((R, G, B), axis=3), np.concatenate((testR, testG, testB), axis=3)
    return train_data, train_labels, test_data, test_labels

def load_imagenet(directory):
    """ load all of imagenet data as flat vector"""
    path_train, path_val = directory + '/ILSVRC2012_img_train', directory + '/ILSVRC2012_img_val'
    train_labels = os.listdir(path_train)
    train_data = []
    for label in train_labels:
        imgs_path = os.path.join(path_train, label)
        imgs = os.listdir(imgs_path)
        for img_name in imgs:
            img_path = os.path.join(imgs_path, img_name)
            img = cv2.imread(img_path)
            b, g, r = cv2.split(img)
            img = cv2.merge([r,g,b]).reshape(-1, 64, 64, 3)
            train_data.append(img)
            train_labels.append(label)
    train_data = np.concatenate(train_data)
    train_labels = np.array(train_labels, dtype='str')
    
    test_labels = os.listdir(path_val)
    test_data = []
    for label in test_labels:
        imgs_path = os.path.join(path_val, label)
        for img_name in imgs:
            img_path = os.path.join(imgs_path, img_name)
            img = cv2.imread(img_path)
            b, g, r = cv2.split(img)
            img = cv2.merge([r,g,b]).reshape(-1, 64, 64, 3)
            test_data.append(img)
            test_labels.append(label)
    test_data = np.concatenate(test_data)
    test_labels = np.array(test_labels, dtype='str')
    
    _, train_labels = np.unique(train_labels, return_inverse=True)
    _, test_labels = np.unique(test_labels, return_inverse=True)
    
    del r, g, b, imgs_path, img_name, img, imgs
    
    return train_data, train_labels, test_data, test_labels

def load_tiny_imagenet(directory):
    """ load all of imagenet data as flat vector"""
    path_train, path_val, path_test = directory + '/train', directory + '/val', directory + '/test'
    labels = os.listdir(path_train)
    train_data = []
    train_labels = []
    for label in labels:
        imgs_path = os.path.join(path_train, label, 'images')
        imgs = os.listdir(imgs_path)
        for img_name in imgs:
            img_path = os.path.join(imgs_path, img_name)
            img = cv2.imread(img_path)
            b, g, r = cv2.split(img)
            img = cv2.merge([r,g,b]).reshape(-1, 64, 64, 3)
            train_data.append(img)
            train_labels.append(label)
    train_data = np.concatenate(train_data)
    train_labels = np.array(train_labels, dtype='str')
    
    test_data = []
    test_labels = []
    with open(path_val+'/val_annotations.txt', 'r') as f:
        val_annotations = [line.strip().split('\t') for line in f]
    val_annotations = np.array(val_annotations)
    imgs_path = os.path.join(path_val, 'images')
    imgs = os.listdir(imgs_path)
    for img_name in imgs:
        img_path = os.path.join(imgs_path, img_name)
        img = cv2.imread(img_path)
        b, g, r = cv2.split(img)
        img = cv2.merge([r,g,b]).reshape(-1, 64, 64, 3)
        test_data.append(img)
        label = val_annotations[val_annotations[:, 0] == img_name, 1].astype('U9')
        test_labels.append(label)
    test_data = np.concatenate(test_data)
    test_labels = np.concatenate(test_labels)
    test_labels = np.array(test_labels, dtype='str')
    
    _, train_labels = np.unique(train_labels, return_inverse=True)
    _, test_labels = np.unique(test_labels, return_inverse=True)
    
    del r, g, b, label, labels, imgs_path, img_name, img, imgs, val_annotations
    
    return train_data, train_labels, test_data, test_labels


def crop_tiny_image_generator(batch_size=64, is_cluster=False):
    def random_crop(img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y + dy), x:(x + dx), :]

    def random_crop_generator(batches, crop_length):
        """Take as input a Keras ImageGen (Iterator) and generate random
        crops from the image batches generated by the original iterator.
        """
        while True:
            batch_x, batch_y = next(batches)
            batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
            for i in range(batch_x.shape[0]):
                batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
            yield (batch_crops, batch_y)

    # data load
    train_data, train_labels, test_data, test_labels = load_tiny_imagenet("datasets/tiny-imagenet-200")
    train_data, test_data = (train_data.astype('float32') - 128.0) / 128.0, (
                test_data.astype('float32') - 128.0) / 128.0
    train_labels, test_labels = to_categorical(train_labels, 200), to_categorical(test_labels, 200)

    # sample data
    if is_cluster:
        with open("10_kmeans_model.sav", 'rb') as fo:
            cluster = pickle.load(fo)
        test_for_clus = test_data.reshape(-1, 12288)
        cluster_pred = cluster.predict(test_for_clus)
        train_data, train_labels = train_data[cluster.labels_ == 4], train_labels[cluster.labels_ == 4]
        test_data, test_labels = test_data[cluster_pred == 4], test_labels[cluster_pred == 4]
        train_data, train_labels = train_data[:10000], train_labels[:10000]
        test_data, test_labels = test_data[:1000], test_labels[:1000]
        del test_for_clus
    else:
        idx = np.random.choice(len(train_data), 10000)
        train_data, train_labels = train_data[idx], train_labels[idx]
        idx = np.random.choice(len(test_data), 1000)
        test_data, test_labels = test_data[idx], test_labels[idx]

    # make generator
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(train_data, train_labels, batch_size=batch_size)
    test_generator = test_datagen.flow(test_data, test_labels, batch_size=batch_size)

    train_crop_generator = random_crop_generator(train_generator, 56)
    test_crop_generator = random_crop_generator(test_generator, 56)

    return train_crop_generator, test_crop_generator

