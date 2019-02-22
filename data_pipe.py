import pickle
import os
import cv2
import numpy as np

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

















