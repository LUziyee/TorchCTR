# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         basemodel
# Description:
# Author:       路子野
# Date:         2020/5/26
#-------------------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader
from ..layers.base import PredictionLayer
from ..inputs import creatEmbeddingMatrix,SparseFeat,DenseFeat
from ..utils import TorchCtrData,collate_fn
from tqdm import tqdm

class BaseModel(torch.nn.Module):
    """


    """
    def __init__(self,module_columns,task="binary",init_std=0.001,learning_rate=0.001):
        """

        :param module_columns:2D list include each module's feature object list, which include SparseFeat objects and DenseFeat objects
        :param task:
        :param init_std:
        :param learning_rate:
        """
        super().__init__()

        """
        设计思路:
            1.模型有多个组件,比如deepfm就包含fm和deep两个部分,有些模型可能有三个甚至四个组件，不同组件输入的特征可能是不同的，因此使用module_columns
            参数来传入每个组件所需要的特征，因此是一个2D list,里面每个list存储的是自定义的feat对象。
            2.用户传入的module_columns可能是乱序的,因此使用self.sortColumns()来排序特征，使得每个组件的特征排列都是类别特征在前
            3.self.sparse_feats用来记录unique feat objects,用来生成embedding_dict，这里的设定是考虑不同组件如果使用同一个特征的话是共享embedding matrix
            的。如果不想共享embedding,那么需要用户在使用时对特征做一下命名区分，比如userid,改成userid_module1,user_id_module2
        """

        self.sparse_feats = set()   #记录所有的类别特征
        self.module_columns = [self.sortColumns(module) for module in module_columns] #排序,使得每个组件的特征排列都是类别特征在前
        self.sparse_feats = list(self.sparse_feats) #把去重后的类别特征转换成数组


        self.learning_rate = learning_rate
        self.task = task
        self.embedding_dict = creatEmbeddingMatrix(self.sparse_feats,init_std)

        self.out = PredictionLayer(self.task)

    def compile(self,optimizer,loss,metrics):
        """

        :param optimizer: String name of optimizer
        :param loss: String name of objective function
        :param metrics: List include metrics name which will be evaluated by model during training and testing.
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
        这里只是给出统一形式的训练过程，至于数据的各块数据的数据，在各个高阶Model的forward()里面完成
        :param x: dict {feat_name:list}
        :param y: list with shape ``(data_len,)``
        :param batch_size: Integer
        :param epochs: Integer
        :param verbose: Integer
        :param shuffle: bool
        :return:
        """
        self.batch_size = batch_size
        self.epochs = epochs

        list_x = []
        for module in self.module_columns:
            for feat in module:
                list_x.append(x[feat.name].tolist())
        ctr_data = TorchCtrData(list_x,y)
        dataloader = DataLoader(ctr_data,batch_size=self.batch_size,shuffle=shuffle,collate_fn=collate_fn,drop_last=False)

        model = self.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            batchs = 0
            for tensor_x,tensor_y in tqdm(dataloader,ncols=80,desc="{} epoch train".format(epoch)):
                output = model(tensor_x)
                batch_loss = self.loss(tensor_x,tensor_y)
                model.zero_grad()
                batch_loss.backward()
                self.optim.step()
                epoch_loss += batch_loss.item()
                batchs += 1
            epoch_mean_loss = epoch_loss/batchs
            print("epoch {}/{},trian loss:{}".format(epoch,self.epochs,epoch_mean_loss))


    def predict(self):
        pass

    def sortColumns(self,columns):
        sparse,dense = [],[]
        for feat in columns:
            if isinstance(feat,SparseFeat):
                sparse.append(feat)
                self.sparse_feats.add(feat)
            else:
                dense.append(feat)
        return sparse+dense























