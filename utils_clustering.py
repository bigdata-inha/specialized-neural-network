import cv2
import numpy as np
import pathlib
from sklearn.cluster import KMeans

def data_to_BGR_image(single_data):
    img_R = single_data[1024].reshape(32, 32, 1)
    img_G = single_data[1024:2048].reshape(32, 32, 1)
    img_B = single_data[2048:].reshape(32, 32, 1)
    image = np.concatenate((img_B, img_G, img_R), axis=2)
    return image


def to_hist(full_data):
    R, G, B = full_data[:, :1024], full_data[:, 1024:2048], full_data[:, 2048:]
    hist_R = np.apply_along_axis(lambda x: np.bincount(x, minlength=256), axis=1, arr=R)
    hist_G = np.apply_along_axis(lambda x: np.bincount(x, minlength=256), axis=1, arr=G)
    hist_B = np.apply_along_axis(lambda x: np.bincount(x, minlength=256), axis=1, arr=B)
    hist_full = np.concatenate((hist_R, hist_G, hist_B), axis=1)
    return hist_full


def save_image_by_cluster(directory, full_data, cluster_labels):
    pathlib.Path(directory).mkdir(exist_ok=True)
    labels_list = np.unique(cluster_labels)
    for i in range(len(labels_list)):
        pathlib.Path(directory + '/cluster%d' % i).mkdir(exist_ok=True)
        cluster_i = full_data[cluster_labels == i]
        img_idx = 0
        for row in cluster_i:
            img = data_to_BGR_image(row)
            cv2.imwrite(directory + '/cluster%d/img%d.jpg' % (i, img_idx), img)
            img_idx = img_idx + 1
    return


def make_cluster(full_data, k=10, cluster_way=1, random_state=1, is_save_as_img=False):
    if cluster_way == 1:    # cluster with RGB values
        cluster_way = 'RGB'
        model = KMeans(n_clusters=k, random_state=random_state).fit(full_data)
    elif cluster_way == 2:  # cluster with RGB histogram
        cluster_way = 'hist'
        full_hist_data = to_hist(full_data)
        model = KMeans(n_clusters=k, random_state=random_state).fit(full_hist_data)
    if is_save_as_img:
        cluster_labels = model.labels_
        directory = 'cluster-result/%d-means-%s-cluster' % (k, cluster_way)
        save_image_by_cluster(directory, full_data, cluster_labels)
    return model
    

