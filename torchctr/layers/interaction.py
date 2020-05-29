# -*- coding: utf-8 -*-#
"""
Name:         interaction
Description:
Author:       路子野
Date:         2020/5/26
"""

import torch
import torch.nn as nn
from torchctr.layers.activation import activation_layer
from torchctr.layers.base import DNN
import itertools


class FM(torch.nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
     Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """
    def __init__(self):
        super().__init__()

    def forward(self,inputs):
        """

        :param inputs: 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
        :return: 2D tensor with shape: ``(batch_size, 1)``.
        """
        square_of_sum = torch.pow(torch.sum(inputs,dim=1,keepdim=True),2)  #shape: (batch_size,1,embedding_size)
        sum_of_square = torch.sum(torch.pow(inputs,2),dim=1,keepdim=True)  #shape: (batch_size,1,embedding_size)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5*torch.sum(cross_term,dim=2,keepdim=False)
        return cross_term


class CrossNet(torch.nn.Module):
    """
    The Cross Network part of Deep&Cross Network model, which leans both low and high degree cross feature.

    Input shape:
        - 2D tensor with shape: ```(batch,input_dim)``` input_dim = filed*module+dense
    Output shape:
        - 2D tensor with shape: ```(batch,input_dim)```

    References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
    """
    def __init__(self,cross_layer,input_dim,init_method,init_std):
        """

        :param cross_layer:
        """
        super().__init__()

        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.empty(input_dim,1)) for i in range(cross_layer)])
        #注意这里bias的维度，和下面forward的写法有关系
        self.bias = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,input_dim)) for i in range(cross_layer)])

        if init_method=="xavier_normal":
            for weight,bias in zip(self.weights,self.bias):
                torch.nn.init.xavier_normal_(weight)
                torch.nn.init.xavier_normal_(bias)
        else:
            for weight,bias in zip(self.weights,self.bias):
                torch.nn.init.normal_(weight,mean=0,std=init_std)
                torch.nn.init.normal_(bias,mean=0,std=init_std)

    def forward(self,x):
        """

        :param x: tensor, shape=(batch,input_dim)
        :return:
        """
        x_0 = x
        x_t = x
        for w,b in zip(self.weights,self.bias):
            x_t_ = x_t.mm(w)            #(batch,1)
            x_t = x_t_*x_0+b+x_t         #(batch,input_dim)  注意这里的维度匹配
        return x_t


class BiInteractionPooling(torch.nn.Module):
    """Bi-Interaction Layer used in Neural FM,compress the
     pairwise element-wise product of features into one single vector.

      Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size,embedding_size)``.

      References
        - [He X, Chua T S. Neural factorization machines for sparse predictive analytics[C]//Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 2017: 355-364.](http://arxiv.org/abs/1708.05027)
    """
    def __init__(self):
        super().__init__()

    def forward(self,x):
        """

        :param x: 3D tensor with shape (batch,filed,embedding_dim)
        :return: 2D tensor with shape: (batch_size,embedding_size).
        """
        square_of_sum = torch.pow(torch.sum(x,dim=1),2)
        sum_of_square = torch.sum(torch.pow(x,2),dim=1)
        pooling = 0.5*(square_of_sum-sum_of_square)
        return pooling


class GMF(torch.nn.Module):
    """
    GMF layer is used in NeuralCF, do element-wise product to imitate matrix factorization

    Input shape:
        - 3D tensor with shape (batch,2,embedding_dim) user vector and item vector

    Output shape:
        - 2D tensor with shape (batch,embedding_dim)

    Reference:
        - He, Xiangnan, et al. “Neural Collaborative Filtering.” WWW ’17 Proceedings of the 26th International Conference on World Wide Web, 2017, pp. 173–182.
    """
    def __init__(self):
        super().__init__()

    def forward(self,x):
        """
        :param x: 3D tensor with shape (batch,2,embedding_dim) user vector and item vector
        :return: 2D tensor with shape (batch,embedding_dim)
        """
        user_embedding = x[:,0,:]  #(batch,embedding_dim)
        item_embedding = x[:,1,:]  #(batch,embedding_dim)
        return user_embedding*item_embedding


class AttentionBasedPooling(torch.nn.Module):
    """Attention-based Pooling Layer used in Attention FM, used to determine
    the importance of interactions between different feature

      Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size,1)``.

      References
        - Xiao, Jun, et al. “Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks.” Twenty-Sixth International Joint Conference on Artificial Intelligence, 2017, pp. 3119–3125.
    """
    def __init__(self,embedding_dim,hidden_units,activation,dropout_rate,init_std):
        super().__init__()

        self.dnn = DNN(input_dim=embedding_dim,
                       hidden_units=hidden_units,
                       activation=activation,
                       dropout_rate=dropout_rate,
                       init_std=init_std)

        self.socre = torch.nn.Linear(hidden_units[-1],1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self,x):
        """
        :param x: A 3D tensor with shape:(batch_size,field,embedding_dim)``.
        :return: 2D tensor with shape: ``(batch,embedding_dim)``.
        """
        filed_list = [i for i in range(x.shape[1])]
        cross_list = []
        for i,j in itertools.combinations(filed_list,2):
            cross_list.append(x[:,[i],:]*x[:,[j],:])
        cross_x = torch.cat(cross_list,dim=1)  #(batch_size,C^filed_2,embedding_dim)
        attention_weight = self.softmax(self.socre(self.dnn(cross_x)).squeeze()).unsqueeze(dim=2) #(batch,C^filed_2,1)
        pool = torch.sum(cross_x*attention_weight,dim=1,keepdim=False)
        afm = torch.sum(pool,dim=1,keepdim=True)
        return afm





