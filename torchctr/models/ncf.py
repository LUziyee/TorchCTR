# -*- coding: utf-8 -*-#
"""
Name:         ncf
Author:       路子野
Date:         2020/5/29
"""

import torch
import torch.nn as nn
from torchctr.layers.base import DNN
from torchctr.layers.interaction import GMF
from torchctr.models.basemodel import BaseModel
from torchctr.inputs import SparseFeat

class NCF(BaseModel):
    """
    Instantiates the Neural Collaborative Filtering architecture.
    """
    def __init__(self,module_columns_dict,hidden_units=[256,128], task="binary",
                 init_std=0.0001,dropout_rate=0,activation="relu"):
        """

        :param module_columns_dict:
        :param hidden_units:
        :param task:
        :param init_std:
        :param dropout_rate:
        :param activation:
        """
        self.module_columns = []  # 存储所有组件的 特征对象 ，tips:特征对象是有重复的
        try:
            # 把各个组件的特征对象放进list去
            self.module_columns.append(module_columns_dict["gmf"])
            self.module_columns.append(module_columns_dict['deep'])
        except:
            raise ValueError("the module's name is wrong")

        super().__init__(module_columns=self.module_columns,
                         init_std=init_std,
                         task=task)

        if not hidden_units:
            raise ValueError("hidden_unit can't be empty")

        deep_input_dim = self._getInputDim(1)

        self.dnn = DNN(input_dim=deep_input_dim,
                       hidden_units=hidden_units,
                       dropout_rate=dropout_rate,
                       init_std=init_std,
                       activation=activation)
        self.outer = torch.nn.Linear(self.module_columns[0][0].embedding_dim+hidden_units[-1], 1)
        self.gmf = GMF()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        gmf_x = x[:,:len(self.module_columns[0])]
        deep_x = x[:,len(self.module_columns[0]):]

        gmf_input = self._get3Dtensor(gmf_x,0)
        deep_input = self._get2Dtensor(deep_x,1)

        gmf,deep = self.gmf(gmf_input),self.dnn(deep_input)
        output = self.sigmoid(self.outer(torch.cat([gmf,deep],dim=1)))
        return output