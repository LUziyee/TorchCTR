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


class Cross(torch.nn.Module):
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
        :return:
        """
        square_of_sum = torch.pow(torch.sum(x,dim=1),2)
        sum_of_square = torch.sum(torch.pow(x,2),dim=1)
        pooling = 0.5*(square_of_sum-sum_of_square)
        return pooling

