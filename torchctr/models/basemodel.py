# -*- coding: utf-8 -*-#
"""
Name:         basemodel
Description:
Author:       路子野
Date:         2020/5/26
"""

import torch
from torch.utils.data import DataLoader
from torchctr.layers.base import PredictionLayer
from torchctr.inputs import creatEmbeddingMatrix, SparseFeat, DenseFeat
from torchctr.utils import TorchCtrData, collate_fn
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score


class BaseModel(torch.nn.Module):
    """


    """

    def __init__(self, module_columns, init_std, learning_rate, task):
        """

        :param module_columns:2D list include each module's feature object list, which include SparseFeat objects and DenseFeat objects
        :param init_std: float, used to initialize layer weight and embedding weight
        :param learning_rate: float,
        :param task: string,
        """
        super().__init__()

        """
        设计思路:
            1.模型有多个组件,比如deepfm就包含fm和deep两个部分,有些模型可能有三个甚至四个组件，不同组件输入的特征可能是不同的，因此使用module_columns
            参数来传入每个组件所需要的特征，因此是一个2D list,里面每个list存储的是自定义的feat对象。
            
            2.用户传入的module_columns可能是乱序的,因此使用self.sortColumns()来排序特征，使得每个组件的特征排列都是类别特征在前
            
            3.self.sparse_feats用来记录unique feat objects,用来生成embedding_dict，这里的设定是考虑不同组件如果使用同一个特征的话是共享embedding matrix
            的。如果不想共享embedding,那么需要用户在使用时对特征做一下命名区分，比如userid,改成userid_module1,user_id_module2
            
            4.只有顶层模型的参数才设默认值，BaseModel和其他组件的参数都不设默认值，都需要靠顶层模型来传递参数，这样才不会混乱，当然，BaseModel的fit函数等需要
            设置默认值，因为这些是用户直接只用的api，所以需要默认值。
        """

        self.sparse_feats = set()  # 记录所有的类别特征
        self.module_columns = [self.__sortColumns(module) for module in module_columns]  # 排序,使得每个组件的特征排列都是类别特征在前
        self.sparse_feats = list(self.sparse_feats)  # 把去重后的类别特征转换成数组

        self.learning_rate = learning_rate
        self.task = task
        self.embedding_dict = creatEmbeddingMatrix(self.sparse_feats, init_std)

        self.out = PredictionLayer(self.task)

    def compile(self, optimizer, loss, metrics=[]):
        """

        :param optimizer: String, name of optimizer, now implement ["sgd","adam"]
        :param loss: String, name of objective function, now implement ["binary_crossentropy","multi_crossentropy"]
        :param metrics: List, include metrics name which will be evaluated by model during training and testing.
        :return: None
        """
        self.optim = self.__getOptim(optimizer)
        self.loss = self.__getLoss(loss)
        self.metrics = metrics

    def __getOptim(self, optimizer):
        if optimizer.lower() == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif optimizer.lower() == "adam":
            optim = torch.optim.Adam(self.parameters())
        else:
            raise NotImplementedError
        return optim

    def __getLoss(self, loss):
        if loss.lower() == "binary_crossentropy":
            loss = torch.nn.BCELoss()
        elif loss.lower() == "multi_crossentropy":
            loss = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        return loss

    def fit(self, x, y, batch_size=32, epochs=10, verbose=1, shuffle=True):
        """
        这里只是给出统一形式的训练过程，至于数据的各块数据的数据，在各个高阶Model的forward()里面完成
        :param x: dict {feat_name:list}
        :param y: list with shape ``(data_len,1)``
        :param batch_size: Integer
        :param epochs: Integer
        :param verbose: Integer
        :param shuffle: bool
        :return:
        """

        """
        思路：
            1.fit()中x传入的字典数据是不重复的，而module_columns里面的特征字段是有重复的
            2.遍历module_columns里面的每个特征对象，存储在list_x中
        """
        self.batch_size = batch_size
        self.epochs = epochs

        list_x = []
        for module in self.module_columns:
            for feat in module:
                list_x.append(x[feat.name].tolist())
        x = np.array(list_x) #（filed*module,data_len),每个组件需要的特征 x 组件个数
        y = np.array(y) #(data_len,1) 注意这里y的shape容易出bug，
        ctr_data = TorchCtrData(x, y)
        dataloader = DataLoader(ctr_data, batch_size=self.batch_size, shuffle=shuffle, collate_fn=collate_fn,
                                drop_last=False)

        model = self.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            batchs = 0
            y_hat = []
            y = []
            for tensor_x, tensor_y in dataloader:
                output = model(tensor_x)
                batch_loss = self.loss(output, tensor_y.float())
                model.zero_grad()
                batch_loss.backward()
                self.optim.step()
                epoch_loss += batch_loss.item()
                batchs += 1
                y_hat.extend(output.detach().numpy())
                y.extend(tensor_y.numpy())
            epoch_mean_loss = epoch_loss / batchs
            metric_result = []
            for metric in self.metrics:
                metric_result.append(self.__metric(metric,y_hat,y))
            print("epoch {}/{}".format(epoch+1,self.epochs))
            print("trian loss:{:.5f}".format(epoch_mean_loss),end=" ")
            for metric,result in zip(self.metrics,metric_result):
                print("{}:{:.5f}".format(metric,result),end=" ")
            print()

    def predict(self):
        pass

    def __sortColumns(self, columns):
        sparse, dense = [], []
        for feat in columns:
            if isinstance(feat, SparseFeat):
                sparse.append(feat)
                self.sparse_feats.add(feat)
            else:
                dense.append(feat)
        return sparse + dense

    def __metric(self,metric,y_hat,y):
        """

        :param metric: str
        :return:
        """
        if metric.lower()=="auc":
            return roc_auc_score(y,y_hat)
        elif metric.lower()=="f1-score":
            return f1_score(y_hat,y)
        else:
            raise NotImplementedError

    def _getInputDim(self,i):
        module = self.module_columns[i]
        input_dim = 0
        for feat in module:
            if isinstance(feat,SparseFeat):
                input_dim += feat.embedding_dim
            else:
                input_dim += feat.dim
        return input_dim

