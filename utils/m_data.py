import numpy as np
import scipy.misc as im
import os
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import cv2
import imageio

def get_mstar_data(stage, width=128, height=128, crop_size=128, aug=False):
    data_dir = "/home/hhq/PyCIL-master/MSTAR-10/train/" if stage == "train" else "/home/hhq/PyCIL-master/MSTAR-10/test/" if stage == "test" else None
    print("------ " + stage + " ------")
    # sub_dir = ["2S1", "BMP2", "BRDM_2", "BTR60", "BTR70", "D7", "T62", "T72", "ZIL131", "ZSU_23_4"]
    # sub_dir = ["ZIL131", "D7", "ZSU_23_4", "T72", "BMP2", "BRDM_2", "T62", "2S1", "BTR70", "BTR60"]
    sub_dir = ["ZIL131", "D7", "ZSU_23_4", "btr70", "t72", "bmp2", "BRDM_2", "T62", "BTR_60", "2S1"]
    # sub_dir = ["ZIL131", "D7", "ZSU_23_4", "btr70", "t72", "bmp2", "BRDM_2", "T62", "BTR_60", "2S1", "Tanker", "Cargo"]
    X = []
    y = []

    for i in range(len(sub_dir)):
        if i < 10:
            tmp_dir = data_dir + sub_dir[i] + "/"
            img_idx = [x for x in os.listdir(tmp_dir) if x.endswith(".jpeg")]
            print(sub_dir[i], len(img_idx))
            y += [i] * len(img_idx)
            for j in range(len(img_idx)):
                img = resize(imageio.imread((tmp_dir + img_idx[j])), [height, width])
                img = img[(height - crop_size) // 2 : height - (height - crop_size) // 2, \
                      (width - crop_size) // 2: width - (width - crop_size) // 2]
                X.append(img)
        else:
            tmp_dir = data_dir + sub_dir[i] + "/"
            img_idx = [x for x in os.listdir(tmp_dir) if x.endswith(".jpeg")]
            print(sub_dir[i], len(img_idx))
            y += [i] * len(img_idx)
            for j in range(len(img_idx)):
                img = resize(imageio.imread((tmp_dir + img_idx[j])), [height, width])
                img = img[(height - crop_size) // 2 : height - (height - crop_size) // 2, \
                      (width - crop_size) // 2: width - (width - crop_size) // 2]
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                X.append(img)
    return np.asarray(X), np.asarray(y) # 2746,96,96 //  2746
    # return X, y

def data_shuffle(X, y, seed=0):
    data = np.hstack([X, y[:, np.newaxis]])
    np.random.shuffle(data)
    return data[:, :-1], data[:, -1]

def one_hot(y_train, y_test):
    one_hot_trans = OneHotEncoder().fit(y_train[:, np.newaxis])
    return one_hot_trans.transform(y_train[:, np.newaxis]).toarray(), one_hot_trans.transform(y_test[:, np.newaxis]).toarray()

def mean_wise(X):
    return (X.T - np.mean(X, axis=1)).T

def pca(X_train, X_test, n):
    pca_trans = PCA(n_components=n).fit(X_train)
    return pca_trans.transform(X_train), pca_trans.transform(X_test)

