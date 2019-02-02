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

def save_cluster_result(full_data, cluster_label):
    for i in range(len(cluster_label)):
        pathlib.Path('cluster-result/cluster%d' % i).mkdir(exist_ok=True)
        cluster_i = full_data[cluster_label == i]
        img_idx = 0
        for row in cluster_i:
            img = data_to_BGRimage(row)
            cv2.imwrite('cluster-result/cluster%d/img%d.jpg' % (i, img_idx), img)
            img_idx = img_idx + 1
    return

def to_hist(full_data):
    R, G, B = full_data[:, :1024], full_data[:, 1024:2048], full_data[:, 2048:]
    hist_R = np.apply_along_axis(lambda x: np.bincount(x, minlength=256), axis=1, arr=R)
    hist_G = np.apply_along_axis(lambda x: np.bincount(x, minlength=256), axis=1, arr=G)
    hist_B = np.apply_along_axis(lambda x: np.bincount(x, minlength=256), axis=1, arr=B)
    hist_full = np.concatenate((hist_R, hist_G, hist_B), axis=1)
    return hist_full


import os
import glob
import pathlib
import cv2
import datetime
import numpy as np
from sklearn.cluster import KMeans

if __name__ == "__main__":

    directory_list = glob.glob('cifar-10-batches/data_batch_*')
    data, label = np.array([], dtype='uint8').reshape(0, 3072), np.array([])
    for dir_batch in directory_list:
        partial_data, partial_label = load_data(dir_batch)
        data = np.vstack([data, partial_data])
        label = np.append(label, partial_label)

    # clustering by image RGB values
    #print('model fitting start:', datetime.datetime.now().strftime('%H:%M:%S'))
    #model = KMeans(n_clusters=100, random_state=1).fit(data)
    #print('model fitting end:', datetime.datetime.now().strftime('%H:%M:%S'))

    # clustering by image RGB histogram
    hist_data = to_hist(data)
    print('model fitting start:', datetime.datetime.now().strftime('%H:%M:%S'))
    model = KMeans(n_clusters=100, random_state=1).fit(hist_data)
    print('model fitting end:', datetime.datetime.now().strftime('%H:%M:%S'))
    
    

