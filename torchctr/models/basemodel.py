# -*- coding: utf-8 -*-#
"""
Name:         basemodel
Description:
Author:       路子野
Date:         2020/5/26
"""

import torch
from torch.utils.data import DataLoader
from torchctr.inputs import SparseFeat, DenseFeat
from torchctr.utils import TorchCtrData, collate_fn
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score
import time


class BaseModel(torch.nn.Module):
    """
    The basemodel of all high-order model, which achieve the train and test function
    """
    def __init__(self, module_cols, init_std, task):
        """
        :param module_cols: 2D-List, include each module's feature object list, which include SparseFeat objects and DenseFeat objects
            e.g. [[feat1,feat2,feat3],[feat1,feat3,feat4,feat5,feat6],[feat3]]
        :param init_std: float, used to initialize layer weight and embedding weight
        :param task: string,
        """
        """
        设计思路:
            1.模型有多个组件,比如deepfm就包含fm和deep两个部分,有些模型可能有三个甚至四个组件，不同组件输入的特征可能是不同的，因此使用module_columns
            参数来传入每个组件所需要的特征，因此是一个2D list,里面每个list存储的是自定义的feat对象。
            
            2.用户传入的module_columns可能是乱序的,因此使用self.sortColumns()来排序特征，使得每个组件的特征排列都是类别特征在前
            
            3.self.sparse_feats用来记录unique feat objects,用来生成embedding_dict，这里的设定是考虑不同组件如果使用同一个特征的话是共享embedding matrix
            的。如果不想共享embedding,那么需要用户在使用时对特征做一下命名区分，比如userid,改成userid_module1,user_id_module2
            
            4.只有顶层模型的参数才设默认值，BaseModel和其他组件的参数都不设默认值，都需要靠顶层模型来传递参数，这样才不会混乱，也避免了写代码写忘了的问题，
            当然，BaseModel的fit函数等需要设置默认值，因为这些是用户直接只用的api，所以需要默认值。
        """
        super().__init__()

        self.sparse_feats = set()  # 记录所有的类别特征，在特征排序时会记录下所有的类别特征
        self.module_cols = [self._sortCols(module) for module in module_cols]  # 排序,使得每个组件的特征排列都是类别特征在前
        self.sparse_feats = list(self.sparse_feats)  # 把去重后的类别特征转换成数组，方便后面创建embedding矩阵字典

        self.task = task
        self.embed_dict = self._creatEmbedMatrix(init_std)

    def compile(self, optimizer, loss, metrics=[],learning_rate=0.001,l2_reg=0.0001):
        """
        compile model, instantiation a loss function object and optimizer object and metrics object
        :param optimizer: String, name of optimizer, now implement ["sgd","adam"]
        :param loss: String, name of objective function, now implement ["binary_crossentropy","multi_crossentropy"]
        :param metrics: List, include metrics name which will be evaluated by model during training and testing.
        :return: None
        """
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.optim = self._getOptim(optimizer)
        self.loss = self._getLoss(loss)

    def _getOptim(self, optimizer):
        if optimizer.lower() == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        elif optimizer.lower() == "adam":
            optim = torch.optim.Adam(self.parameters(),lr=self.learning_rate,weight_decay=self.l2_reg)
        else:
            raise NotImplementedError
        return optim

    def _getLoss(self, loss):
        if loss.lower() == "binary_crossentropy":
            loss = torch.nn.BCELoss()
        elif loss.lower() == "multi_crossentropy":
            loss = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        return loss

    def fit(self, x, y, batch_size=32, epochs=10, verbose=1, shuffle=True):
        """
        这里只是给出统一形式的训练过程，至于数据的各块数据的处理，特别是类别特征的embedding，在各个高阶Model的forward()里面完成
        :param x: dict {feat_name:ndarray}
        :param y: list with shape ``(data_len,1)``
        :param batch_size: Integer
        :param epochs: Integer
        :param verbose: Integer, control print interval
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
        for module in self.module_cols:
            for feat in module:
                list_x.append(x[feat.name].tolist())
        x = np.array(list_x) #（filed*module,data_len),每个组件需要的特征 x 组件个数
        y = np.array(y).reshape(-1,1) #保证传入ctr_data的y的shape是(data_len,1)
        ctr_data = TorchCtrData(x, y)
        dataloader = DataLoader(ctr_data,
                                batch_size=self.batch_size,
                                shuffle=shuffle,
                                collate_fn=collate_fn,
                                drop_last=False)

        model = self.train()
        for epoch in range(self.epochs):
            st_time = time.time()
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
                metric_result.append(self._metric(metric, y_hat, y))
            if verbose==0:
                pass
            elif epoch%verbose==0:
                end_time = time.time()
                print("Epoch {}/{}".format(epoch+1,self.epochs))
                print("{}s -".format(int(end_time-st_time)),end=" ")
                print("trian loss: {:.5f} -".format(epoch_mean_loss),end=" ")
                for metric,result in zip(self.metrics,metric_result):
                    print("{}: {:.5f} -".format(metric,result),end=" ")
                print()
            else:
                pass

    def test(self, x, y, batch_size=256, shuffle=True):
        """
        这里实现了统一的test功能
        :param x:
        :param y:
        :param batch_size:
        :param shuffle:
        :return:
        """
        list_x = []
        for module in self.module_cols:
            for feat in module:
                list_x.append(x[feat.name].tolist())
        x = np.array(list_x)  # （filed*module,data_len),每个组件需要的特征 x 组件个数
        y = np.array(y).reshape(-1,1)  # (data_len,1) 注意这里y的shape容易出bug，
        ctr_data = TorchCtrData(x, y)
        dataloader = DataLoader(ctr_data,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=collate_fn,
                                drop_last=False)

        model = self.eval()
        epoch_loss = 0
        batchs = 0
        y_hat = []
        y = []
        for tensor_x, tensor_y in dataloader:
            output = model(tensor_x)
            batch_loss = self.loss(output, tensor_y.float())
            epoch_loss += batch_loss.item()
            batchs += 1
            y_hat.extend(output.detach().numpy())
            y.extend(tensor_y.numpy())
        epoch_mean_loss = epoch_loss / batchs
        metric_result = []
        for metric in self.metrics:
            metric_result.append(self._metric(metric, y_hat, y))
        print()
        print("==========Test Stage========")
        print("test loss:{:.5f}".format(epoch_mean_loss), end=" ")
        for metric, result in zip(self.metrics, metric_result):
            print("{}:{:.5f}".format(metric, result), end=" ")
        print()

    def _sortCols(self, cols):
        """
        1.特征排序，使得离散特征在连续特征前面
        2.把每个离散特征都加到self.sparse_feats里面
        :param columns: List including SparseFeat&DenseFeat
        :return:
        """
        sparse, dense = [], []
        for feat in cols:
            if isinstance(feat, SparseFeat):
                sparse.append(feat)
                self.sparse_feats.add(feat)
            else:
                dense.append(feat)
        return sparse + dense

    def _metric(self, metric, y_hat, y):
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

    def _getInputDim(self, i):
        """
        计算某个组件的输入维度，最典型的应用就是计算dnn的input_dim
        :param i: Integer, denote i-th module, as the index of module_columns
        :return: Integer, input_dim
        """
        module = self.module_cols[i]
        input_dim = 0
        for feat in module:
            if isinstance(feat,SparseFeat):
                input_dim += feat.embed_dim
            else:
                input_dim += feat.dim
        return input_dim

    def _get3Dtensor(self,x,i):
        """
        将类别特征的id在对应的embedding matrix中进行lookup,并进行合并，
        :param x: 2D tensor with shape (batch,filed)
        :param i: Integer, denote i-th module, as the index of module_columns
        :return: 3D tensor with shape (batch,filed,embedding_dim)
        """
        embedding_list = []
        for index,feat in enumerate(self.module_cols[i]):
            feat_id = x[:,[index]].long()  #(batch,1)
            embedding_list.append(self.embed_dict[feat.name](feat_id))  #[(batch,1,embedding_dim)]
        fm_input = torch.cat(embedding_list,dim=1) #(batch,filed,embedding_dim)
        return fm_input

    def _get2Dtensor(self, x, i):
        """
        将类别特征的id在对应的embedding matrix中进行lookup,并进行concat,并将连续特征拼接在后面
        :param x: 2D tensor with shape (batch,filed+dense)
        :param i: Integer, denote i-th module, as the index of module_columns
        :return: 2D tensor with shape (batch,filed*embedding_dim+dense)
        """
        embedding_list = []
        dense_list = []
        for index, feat in enumerate(self.module_cols[i]):
            if isinstance(feat, SparseFeat):
                feat_id = x[:, [index]].long()
                embedding_list.append(self.embed_dict[feat.name](feat_id).squeeze(dim=1))
            else:
                dense_list.append(x[:, [index]].float())
        deep_input = torch.cat(embedding_list + dense_list, dim=1)  # (batch,filed*embedding_dim+dense)
        return deep_input

    def _creatEmbedMatrix(self, init_std):
        """
        generate embedding matrix object for sparse feature
        :param init_std: float
        :return:
        待填的坑：
            1.自定义的初始化方式
        """
        self.embed_dict = torch.nn.ModuleDict()
        for feat in self.sparse_feats:
            if isinstance(feat, SparseFeat):
                self.embed_dict[feat.name] = torch.nn.Embedding(feat.vocab_size, feat.embed_dim)

        for matrix in self.embed_dict.values():
            torch.nn.init.normal_(matrix.weight, mean=0, std=init_std)


