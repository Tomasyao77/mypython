import sys
import os
from config import cfg
import numpy as np
import torch

if __name__ == '__main__':
    # f1 = os.listdir(cfg.dataset.morph2)
    # f2 = os.listdir(cfg.dataset.morph2)
    # print(f1[1000] == f2[1000])

    # g = np.ones((128, 3, 128, 128))
    # print(g.shape)
    # print(g[:, :, :, :-1].shape)
    # print(g[:, :, :, 1:].shape)

    gender_tensor = -torch.ones(2)
    print(gender_tensor.shape)
    gender_tensor[int(0)] *= -1
    # gender_tensor [2] repeat -> [10]
    gender_tensor = gender_tensor.repeat(10,
                                         10 // 2)
    print(gender_tensor.shape)
