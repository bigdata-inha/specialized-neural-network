def load_data(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    data = np.array(dict[b'data'])
    label = np.array(dict[b'labels'])
    return data, label

def data_to_BGRimage(single_data):
    img_R = single_data[:1024].reshape(32, 32, -1)
    img_G = single_data[1024:2048].reshape(32, 32, -1)
    img_B = single_data[2048:].reshape(32, 32, -1)
    image = np.concatenate((img_B, img_G, img_R), axis=2)
    return image

def to_hist(single_data):
    return


import os
import glob
import pathlib
import cv2
import datetime
import numpy as np
from sklearn.cluster import KMeans

if __name__ == "__main__":

    directory_list = glob.glob('cifar-10-batches/data_batch_*')
    data, label = np.array([]).reshape(0, 3072), np.array([])
    for dir_batch in directory_list:
        partial_data, partial_label = load_data(dir_batch)
        data = np.vstack([data, partial_data])
        label = np.append(label, partial_label)

    print('model fitting start:', datetime.datetime.now().strftime('%H:%M:%S'))
    model = KMeans(n_clusters=100, random_state=1).fit(data)
    print('model fitting end:', datetime.datetime.now().strftime('%H:%M:%S'))

    cluster_label = model.labels_
    idx = np.zeros(100)
    for i in range(100):
        pathlib.Path('cifar-10-image/cluster%d' % i).mkdir(exist_ok=True)
        i_cluster = data[cluster_label == i]
        for row in i_cluster:
            img = data_to_BGRimage(row)
            cv2.imwrite('cifar-10-image/cluster%d/img%d.jpg' % (i, idx[i]), img)
            idx[i] = idx[i] + 1


                

