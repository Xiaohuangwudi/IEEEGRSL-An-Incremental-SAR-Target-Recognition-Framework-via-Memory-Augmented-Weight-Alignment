# -*- coding: utf-8 -*-
import matplotlib
# matplotlib.use('agg')
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt

class ProgressiveSampler(object):
    def __init__(self, dataset, max_epoch, train_targets, nums_per_cls):
        self.max_epoch = max_epoch
        self.dataset = dataset      # dataset
        self.train_targets = train_targets
        self.nums_per_cls = nums_per_cls

    def _cal_class_prob(self, q):
        num_pow = list(map(lambda x: pow(x, q), self.nums_per_cls))
        sigma_num_pow = sum(num_pow)
        cls_prob = list(map(lambda x: x/sigma_num_pow, num_pow))
        return cls_prob

    def _cal_pb_prob(self, t):
        p_ib = self._cal_class_prob(q=1)
        p_cb = self._cal_class_prob(q=0)
        p_pb = (1 - t/self.max_epoch) * np.array(p_ib) + (t/self.max_epoch) * np.array(p_cb)

        p_pb /= np.array(self.nums_per_cls)  
        return p_pb.tolist()

    def __call__(self, epoch):

        p_pb = self._cal_pb_prob(t=epoch)
        p_pb = torch.tensor(p_pb, dtype=torch.float)
        samples_weights = p_pb[self.train_targets]  
        sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights))
        # sampler = WeightedRandomSampler(weights=samples_weights, num_samples=1000)
        return sampler, p_pb

    def plot_line(self):
        for i in range(self.max_epoch):
            _, weights = self(i)
            if i % 10 == 9:
                x = range(len(weights))
                plt.plot(x, weights, label="t="+str(i))
        plt.legend()
        plt.title("max epoch="+str(self.max_epoch))
        plt.xlabel("class index sorted by numbers")
        plt.ylabel("weights")
        plt.show()
