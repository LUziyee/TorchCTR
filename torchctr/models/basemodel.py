# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         basemodel
# Description:
# Author:       路子野
# Date:         2020/5/26
#-------------------------------------------------------------------------------

import torch
from ..layers.base import PredictionLayer
from ..inputs import creatEmbeddingMatrix,SparseFeat,DenseFeat

class BaseModel(torch.nn.Module):
    """


    """
    def __init__(self,linear_feature_columns,dnn_feature_columns,task="binary",init_std=0.001,learning_rate=0.001):
        """

        :param linear_feature_columns:list include SparseFeat objects and DenseFeat objects
        :param dnn_feature_columns:list include SparseFeat objects and DenseFeat objects
        :param task:
        :param init_std:
        :param learning_rate:
        """
        super().__init__()
        #保持类别特征在前的顺序，方便后面embedding
        sparse,dense = [],[]
        for feat in linear_feature_columns:
            if isinstance(feat,SparseFeat):
                sparse.append(feat)
            else:
                dense.append(feat)
        self.linear_feature_columns = sparse+dense
        sparse, dense = [], []
        for feat in dnn_feature_columns:
            if isinstance(feat, SparseFeat):
                sparse.append(feat)
            else:
                dense.append(feat)
        self.dnn_feature_columns = sparse+dense
        self.learning_rate = learning_rate
        self.task = task
        self.embedding_dict = creatEmbeddingMatrix(self.dnn_feature_columns,init_std)

        self.out = PredictionLayer(self.task)

    def compile(self,optimizer,loss,metrics):
        """

        :param optimizer: String name of optimizer
        :param loss: String name of objective function
        :param metrics: List include metrics name which will be evaluated by model during traning and testing.
        :return: None
        """
        self.optim = self.__get_optim(optimizer)
        self.loss = self.__get_loss(loss)


    def __get_optim(self,optimizer):
        if optimizer.lower() == "sgd":
            optim = torch.optim.SGD(self.parameters(),lr=self.learning_rate)
        elif optimizer.lower() == "adam":
            optim = torch.optim.Adam(self.parmeters(),lr=self.learning_rate)
        else:
            raise NotImplementedError
        return optim

    def __get_loss(self,loss):
        if loss.lower() == "binary_crossentropy":
            loss = torch.nn.BCEloss()
        elif loss.lower() == "multi_crossentropy":
            loss = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        return loss



    def fit(self,x,y,batch_size,epochs,verbose=1,shuffle=True):
        """

        :param x: dict {feat_name:list}
        :param y: list with shape ``(data_len,)``
        :param batch_size: Integer
        :param epochs: Integer
        :param verbose: Integer
        :param shuffle: bool
        :return:
        """
        list_x = []















    def predict(self):
        pass

























