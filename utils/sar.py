# -*- coding: utf-8 -*-
"""
# @file name  : matar.py
# @author     : huang
# @date       : 2023年2月28日
# @brief      : mstar数据集读取
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import scipy.misc as im
import os
import imageio
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA


# sub_dir = ["ZIL131", "D7", "ZSU_23_4", "btr70", "t72", "bmp2", "BRDM_2", "T62", "BTR_60", "2S1", "Tanker", "Cargo"]


names = {'ZIL131': 0, 'D7': 1, 'ZSU_23_4': 2, 'BTR70': 3, 'T72': 4, 'BMP2': 5, 'BRDM_2': 6, 'T62': 7, 'BTR60': 8,
         '2S1': 9, 'Tanker': 10, 'Cargo': 11}

class SarDataset(Dataset):
    cls_num = 12
    names = tuple([i for i in range(cls_num)])

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []      # 定义list用于存储样本路径、标签
        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label, path_img

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))   # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        for root, dirs, _ in os.walk(self.root_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                # 修改这里，同时支持.jpg和.jpeg扩展名
                img_names = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.jpeg'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.abspath(os.path.join(root, sub_dir, img_name))
                    label = names[sub_dir]
                    self.img_info.append((path_img, label))
        random.shuffle(self.img_info)   # 将数据顺序打乱



class SarLTDataset(SarDataset):
    def __init__(self, root_dir, transform=None, imb_factor=0.01, isTrain=True):
        """
        :param root_dir:
        :param transform:
        :param imb_type:
        :param imb_factor: float, 值越小，数量下降越快,0.1表示最少的类是最多的类的0.1倍，如500：5000
        :param isTrain:
        """
        super(SarLTDataset, self).__init__(root_dir, transform=transform)
        self.imb_factor = imb_factor
        if isTrain:
            self.nums_per_cls = self._get_img_num_per_cls()     # 计算每个类的样本数
            self._select_img()      # 采样获得符合长尾分布的数据量
        else:
            # 非训练状态，可采用均衡数据集测试
            self.nums_per_cls = []
            for n in range(self.cls_num):
                label_list = [label for p, label in self.img_info]  # 获取每个标签
                self.nums_per_cls.append(label_list.count(n))       # 统计每个类别数量

    def _select_img(self):
        """
        根据每个类需要的样本数进行挑选
        :return:
        """
        new_lst = []
        for n, img_num in enumerate(self.nums_per_cls):
            lst_tmp = [info for info in self.img_info if info[1] == n]  # 获取第n类别数据信息
            random.shuffle(lst_tmp)
            lst_tmp = lst_tmp[:img_num]
            new_lst.extend(lst_tmp)
        random.shuffle(new_lst)
        self.img_info = new_lst

    def _get_img_num_per_cls(self):
        """
        依长尾分布计算每个类别应有多少张样本
        :return:
        """
        img_max = len(self.img_info) / self.cls_num
        img_num_per_cls = []
        for cls_idx in range(self.cls_num):
            num = img_max * (self.imb_factor ** (cls_idx / (self.cls_num - 1.0)))  # 列出公式就知道了
            img_num_per_cls.append(int(num))
        return img_num_per_cls



def get_sar_data(stage):
    data_dir = "/home/hhq/SARIL/data/sar/train/" if stage == "train" else "/home/hhq/SARIL/data/sar/test/" if stage == "test" else None
    print("------ " + stage + " ------")
    sub_dir = ['ZIL131', 'D7', 'ZSU_23_4', 'BTR70', 'T72', 'BMP2', 'BRDM_2', 'T62', 'BTR60',
         '2S1', 'Tanker', 'Cargo']
    X = []
    y = []

    for i in range(len(sub_dir)):
        tmp_dir = data_dir + sub_dir[i] + "/"
        img_idx = [x for x in os.listdir(tmp_dir) if x.endswith(".jpg") or x.endswith(".jpeg")]
        print(sub_dir[i], len(img_idx))
        y += [i] * len(img_idx)
        for j in range(len(img_idx)):
            img = resize(imageio.imread(tmp_dir + img_idx[j]), [64, 64,3 ])
            X.append(img)
            # img = resize(imageio.imread(tmp_dir + img_idx[j]), [64, 64])
            # X.append(img)
    return np.asarray(X), np.asarray(y)

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