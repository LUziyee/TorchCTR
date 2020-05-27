# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         utils
# Description:
# Author:       路子野
# Date:         2020/5/27
#-------------------------------------------------------------------------------

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader

class TorchCtrData(Dataset):
    def __init__(self,x,y):
        """
        :param x: ndarray with shape ``(field,date_len)``
        :param y: ndarray with shape ``(data_len,)``
        """
        super().__init__()

        self.x = x.reshape(-1,x.shape[0])
        self.y = y.reshape(-1,1)

    def __getitem__(self, index):
        sample_x = self.x[index]
        sample_y = self.y[index]
        return (sample_x,sample_y)

    def __len__(self):
        return self.x.shape[0]

def collate_fn(batch):
    """
    :param batch: tuple (batch_x,batch_y)
    :return:
    """
    df_x,df_y = batch
    tensor_x = torch.tensor(df_x)
    tensor_y = torch.tensor(df_y)
    return tensor_x,tensor_y

