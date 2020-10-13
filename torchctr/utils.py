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
import time
from sklearn.metrics import f1_score, roc_auc_score


class TorchCtrData(Dataset):
    def __init__(self, x, y):
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
    array_x, array_y = batch[:, :-1], batch[:, -1]
    tensor_x = torch.tensor(array_x)
    tensor_y = torch.tensor(array_y)
    return tensor_x, tensor_y


class Trainer():
    """
    目的是生成一个训练类对象，传入模型和dataloader，然后选择train or train_eval or eval or predict 四种模式
    还应该有日志生成和模型保存的功能，以及early-stopping的功能，能返回best_epoch等
    """

    def __init__(self, params):
        """

        :param params: dict
        """
        self.params = params

    def train(self, model, dataloader, verbose):
        epochs = self.params['epochs']
        lr = self.params['lr']
        loss = self.params['loss']
        optim = self.params['optimizer']
        metrics = self.params['metrics']

        if 'logloss' == loss:
            loss = torch.nn.BCELoss()
        else:
            raise NotImplementedError
        if optim == 'adam':
            optim = torch.optim.Adam(model.parameters(), lr=lr)
        elif optim == 'sgd':
            optim = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise NotImplementedError

        model.train()
        for epoch in range(epochs):
            st_time = time.time()
            epoch_loss = 0
            batchs = 0
            y_hat = []
            y = []
            for tensor_x, tensor_y in dataloader:
                output = model(tensor_x)
                batch_loss = loss(output, tensor_y.float())
                model.zero_grad()
                batch_loss.backward()
                optim.step()
                epoch_loss += batch_loss.item()
                batchs += 1
                y_hat.extend(output.detach().numpy())
                y.extend(tensor_y.numpy())
            epoch_mean_loss = epoch_loss / batchs
            metric_result = []
            for metric in metrics:
                metric_result.append(self.__metric(metric, y_hat, y))
            if verbose == 0:
                pass
            elif epoch % verbose == 0:
                end_time = time.time()
                print("Epoch {}/{}".format(epoch + 1, self.epochs))
                print("{}s -".format(int(end_time - st_time)), end=" ")
                print("train loss: {:.5f} -".format(epoch_mean_loss), end=" ")
                for metric, result in zip(self.metrics, metric_result):
                    print("{}: {:.5f} -".format(metric, result), end=" ")
                print()
            else:
                pass

    def train_eval(self):
        pass

    def eval(self):
        pass

    def predict(self):
        pass

    def __metric(self, metric, y_hat, y):
        """
        :param metric: str
        :return:
        """
        if metric.lower() == "auc":
            return roc_auc_score(y, y_hat)
        elif metric.lower() == "f1-score":
            return f1_score(y_hat, y)
        else:
            raise NotImplementedError
