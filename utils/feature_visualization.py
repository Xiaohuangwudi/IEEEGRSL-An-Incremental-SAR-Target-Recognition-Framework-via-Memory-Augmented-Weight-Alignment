# -*- coding: utf-8 -*-
# @Time    : 2021/4/9 9:48 上午
# @Author  : T'ang Chia-Hsin
# @E-mail  : EkAugust@icloud.com
# @Version : 1.0
import collections
import torch
import numpy as np
# from models.resnet import *
from convs.resnet import *
from utils.aaaa import MSTAR
from torchvision import transforms
from torch.utils.data import Subset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict
import collections

def visualizaiton_features(data, label, ax):
    color_list = ['paleturquoise', 'cornflowerblue', 'thistle', 'orchid', 'lightpink', 'darkred', 'darkorange',
                  'darkkhaki',
                  'darkolivegreen', 'darkseagreen']
    marker_list = ['8', 's', 'p', 'P', '*', 'h', 'H', 'X', 'D', 'd']
    Label_list = ["ZIL131", "D7", "BTR70", "T72", "BMP2", "BRDM2", "T62", "BTR60", "2S1", "ZSU23/4"]
    x_min = np.min(data, 0)
    x_max = np.max(data, 0)
    f2d_norm = (data - x_min) / (x_max - x_min)
    x, y = np.split(f2d_norm, 2, axis=1)
    label_id = list(set(label))
    label_id.sort()
    label_num = []
    for i in label_id:
        label_num.append(label.count(i))

    upper_boundery = 0
    bottom_boundery = 0
    count = 0
    for i in label_num:
        upper_boundery += i
        ax.scatter(x[bottom_boundery:upper_boundery], y[bottom_boundery:upper_boundery], 20, c=color_list[count],
                   alpha=0.5,
                   marker=marker_list[count],
                   label=Label_list[count])
        bottom_boundery += i
        count += 1

    plt.legend(loc='upper left')


def feature_vis(network):
    ft_net_c10 = network(num_classes=13)
    ft_net_c10.load_state_dict(torch.load('E:\incremental learning\codes\EXP_Result\iCaRL2_P_pr/0\TASK_6\model.pkl',
                                          map_location='cpu'), strict=False)
    # ft_net_c10.load_state_dict(
    #     torch.load('E:\incremental learning\PyCIL-master\weight\9\model.pkl', map_location='cpu'),
    #     strict=False)
    # icarl_net_c10 = resnet18(num_classes=10)
    # icarl_net_c10.load_state_dict(
    #     torch.load('E:\incremental learning\PyCIL-master\weight\9\model.pkl', map_location='cpu'))
    # icarl2_net_c10 = resnet18(num_classes=10)
    # icarl2_net_c10.load_state_dict(
    #     torch.load('E:\incremental learning\PyCIL-master\weight\9\model.pkl', map_location='cpu'))


    # ft_net_c10 = resnet18(num_classes=10)
    # state_dict = ft_net_c10.state_dict()
    # weight = torch.load('E:\incremental learning\PyCIL-master\weight\9\model.pkl')
    # ft_net_c10.load_state_dict(weight, strict= False)

    icarl_net_c10 = resnet18(num_classes=10)
    base_weights = torch.load('E:\incremental learning\PyCIL-master\weight\9\model.pkl')
    new_state_dict = OrderedDict()
    for k, v in base_weights.items():
        name = k[8:]
        new_state_dict[name] = v
        icarl_net_c10.load_state_dict(new_state_dict, strict=False)




    icarl2_net_c10 = resnet18(num_classes=10)
    state_dict =  icarl2_net_c10.state_dict()
    weight = torch.load('E:\incremental learning\PyCIL-master\weight\9\model.pkl')
    icarl2_net_c10.load_state_dict(weight, strict=False)


    # loading test set
    VAL_DATA_DIR = './test.txt'
    test_dataset = MSTAR(VAL_DATA_DIR, 88, 2, 5, transform=transforms.ToTensor())
    test_batches = []
    for i in range(2):
        test_batch_indexes = []
        for j in range(i + 1):
            test_indexes = test_dataset.__getBatchIndexes__(j)
            test_batch_indexes += test_indexes

        test_batch = Subset(test_dataset, test_batch_indexes)
        test_batches.append(test_batch)

    test_batch_data = []
    test_batch_label = []

    for i in range(len(test_batches)):
        test_batch_data.append([])
        test_batch_label.append([])
        for j in range(len(test_batches[i])):
            test_batch_data[i].append(test_batches[i][j][0])
            test_batch_label[i].append(test_batches[i][j][1])

    test_batch_data[0] = torch.cat(test_batch_data[0], 0).reshape(len(test_batch_data[0]), 1, 88, 88)
    test_batch_data[1] = torch.cat(test_batch_data[1], 0).reshape(len(test_batch_data[1]), 1, 88, 88)

    # ft_features_c10 = ft_net_c10(test_batch_data[1], features_only=True)
    ft_features_c10 = ft_net_c10(test_batch_data[1])['features']
    label_c10 = test_batch_label[1]
    ft_features_c10 = np.array(ft_features_c10.detach())

    tsne = TSNE(n_components=2)
    tf_features_c10_2d = tsne.fit_transform(ft_features_c10)

    # icarl_features_c10 = icarl_net_c10(test_batch_data[1], features_only=True)
    icarl_features_c10 = icarl_net_c10(test_batch_data[1])['features']
    icarl_features_c10 = np.array(icarl_features_c10.detach())

    tsne = TSNE(n_components=2)
    icarl_features_c10_2d = tsne.fit_transform(icarl_features_c10)

    # icarl2_features_c10 = icarl2_net_c10(test_batch_data[1], features_only=True)
    icarl2_features_c10 = icarl2_net_c10(test_batch_data[1])['features']
    icarl2_features_c10 = np.array(icarl2_features_c10.detach())

    tsne = TSNE(n_components=2)
    icarl2_features_c10_2d = tsne.fit_transform(icarl2_features_c10)

    fig = plt.figure(figsize=(16, 5), tight_layout=True)
    fig.set_canvas(plt.gcf().canvas)
    gs = gridspec.GridSpec(1, 3)

    ax = fig.add_subplot(gs[0, 0])
    visualizaiton_features(tf_features_c10_2d, label_c10, ax)

    ax = fig.add_subplot(gs[0, 1])
    visualizaiton_features(icarl_features_c10_2d, label_c10, ax)

    ax = fig.add_subplot(gs[0, 2])
    visualizaiton_features(icarl2_features_c10_2d, label_c10, ax)

    fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()

    plt.savefig('Features_visualization.pdf')
    plt.show()
    plt.close()
