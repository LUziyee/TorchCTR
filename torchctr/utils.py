# -*- coding: utf-8 -*-#
"""
Name:         utils
Description:
Author:       路子野
Date:         2020/5/27
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class TorchCtrData(Dataset):
    def __init__(self,x,y):
        """
        :param x: ndarray with shape ``(field*module,date_len)``
        :param y: ndarray with shape ``(data_len,)``
        """
        super().__init__()

        """
        这里有一个坑，关于reshape还是T的问题。
        """
        self.x = x.T
        self.y = y

    def __getitem__(self, index):
        sample_x = self.x[index].tolist()
        sample_y = self.y[index].tolist()
        sample_x.extend(sample_y)
        return sample_x

    def __len__(self):
        return self.x.shape[0]

def collate_fn(batch):
    """
    :param batch: tuple (batch_x,batch_y)
    :return:
    """
    batch = np.array(batch)
    array_x,array_y = batch[:,:-1],batch[:,-1]
    tensor_x = torch.tensor(array_x)
    tensor_y = torch.tensor(array_y)
    return tensor_x,tensor_y

