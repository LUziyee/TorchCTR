# -*- coding: utf-8 -*-#
"""
Name:         dcn
Author:       路子野
Date:         2020/5/28
"""

from torchctr.layers.base import DNN
from torchctr.layers.interaction import Cross
from torchctr.layers.activation import activation_layer
import torch
import torch.nn as nn
from torchctr.models.basemodel import BaseModel
from torchctr.inputs import SparseFeat,DenseFeat,creatEmbeddingMatrix
import warnings
warnings.filterwarnings("ignore")

class DCN(BaseModel):
    """

    """
    def __init__(self,module_columns_dict,init_std=0.0001, learning_rate=0.001,
                 task="binary",cross_layer=3,hidden_units=[128,64],dropout_rate=0.5,
                 activation="relu",init_method="normal"):
        """

        :param module_columns_dict:
        :param init_std:
        :param learning_rate:
        :param task:
        :param cross_layer:
        :param hidden_units:
        :param dropout_rate:
        :param activation:
        """
        module_columns = []
        try:
            module_columns.append(module_columns_dict["cross"])
            module_columns.append(module_columns_dict["deep"])
        except:
            raise ValueError("the module's name is wrong")

        super().__init__(module_columns=module_columns,
                         init_std=init_std,
                         learning_rate=learning_rate,
                         task=task)

        cross_input_dim = self._getInputDim(0)
        dnn_input_dim = self._getInputDim(1)


        self.cross_net = Cross(cross_layer=cross_layer,
                               input_dim=cross_input_dim,
                               init_std=init_std,
                               init_method=init_method)

        self.dnn = DNN(input_dim=dnn_input_dim,
                       hidden_units=hidden_units,
                       dropout_rate=dropout_rate,
                       init_std=init_std,
                       activation=activation)

        self.outer = torch.nn.Linear(cross_input_dim+hidden_units[-1],1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        """

        :param x: tensor, with shape (batch,filed*module)
        :return:
        """
        cross_x = x[:,:len(self.module_columns[0])]
        deep_x = x[:,len(self.module_columns[1]):]

        cross_embedding_list = []
        cross_dense_list = []
        for index, feat in enumerate(self.module_columns[0]):
            if isinstance(feat, SparseFeat):
                feat_id = cross_x[:, [index]].long()
                cross_embedding_list.append(self.embedding_dict[feat.name](feat_id).squeeze(dim=1))
            else:
                cross_dense_list.append(cross_x[:, [index]].float())
        cross_input = torch.cat(cross_embedding_list + cross_dense_list, dim=1)  # (batch,filed*embedding_dim+dense)
        cross = self.cross_net(cross_input)  #(batch,input_dim)

        deep_embedding_list = []
        deep_dense_list = []
        for index, feat in enumerate(self.module_columns[1]):
            if isinstance(feat, SparseFeat):
                feat_id = deep_x[:, [index]].long()
                deep_embedding_list.append(self.embedding_dict[feat.name](feat_id).squeeze(dim=1))
            else:
                deep_dense_list.append(deep_x[:, [index]].float())
        deep_input = torch.cat(deep_embedding_list + deep_dense_list, dim=1)  # (batch,filed*embedding_dim+dense)
        deep = self.dnn(deep_input) #(batch,hidden_units[-1])
        out = self.sigmoid(self.outer(torch.cat([cross,deep],dim=1)))

        return out


