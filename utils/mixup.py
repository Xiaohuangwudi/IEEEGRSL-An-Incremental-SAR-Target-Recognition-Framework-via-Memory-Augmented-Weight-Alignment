import numpy as np
import torch

import torch.nn.functional as F
import torch.nn as nn


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)  # log_p 向量
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))  # Q向量
        loss = (-weight * log_prob).sum(dim=-1).mean()  # log_p * Q 再相加
        return loss


def mixup_data(x, y, alpha=1.0, device=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    path_1 = r'E:\incremental learning\PyCIL-master\MSTAR-10\train\2S1\hb19377.jpeg'
    path_2 = r'E:\incremental learning\PyCIL-master\MSTAR-10\train\BMP2\hb03787.jpeg'

    img_1 = cv2.imread(path_1)
    img_2 = cv2.imread(path_2)
    print(img_1.size)
    img_1 = cv2.resize(img_1, (88, 88))
    img_2 = cv2.resize(img_2, (88, 88))

    alpha = 1.
    figsize = 15
    plt.figure(figsize=(int(figsize), int(figsize)))
    for i in range(1, 10):
        # lam = i * 0.1
        lam = np.random.beta(alpha, alpha)
        im_mixup = (img_1 * lam + img_2 * (1 - lam)).astype(np.uint8)
        im_mixup = cv2.cvtColor(im_mixup, cv2.COLOR_BGR2RGB)
        plt.subplot(3, 3, i)
        plt.title("lambda_{:.2f}".format(lam))
        plt.imshow(im_mixup)
    plt.show()








