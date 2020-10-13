# -*- coding:utf-8 -*-
"""
@Time:19:03
@Author:LuZiye
"""

import torch
from torchctr.models.basemodel import BaseModel
from torchctr.layers.base import DNN
from torchctr.inputs import SparseFeat,DenseFeat
from torchctr.layers.interaction import FM


class DeepFM(BaseModel):
    def __init__(self, module_cols_dict, hidden_units=[256,128,64], task="binary",
                 init_std=0.0001,dropout_rate=0,activation="relu"):
        """
        :param module_cols_dict: dict, {module_name:[sparsefeat1,sparsefeat2,densefeat1,...]}
        :param hidden_units:list, default=[256,128,64]
        :param task: string,
        :param init_std: float, used to initialize layer weight and embedding weight
        """
        self.module_cols = []  #存储所有组件的 特征对象 ，tips:特征对象是有重复的 2D-List
        try:
            #把各个组件的特征对象放进list去
            self.module_columns.append(module_cols_dict["fm"])
            self.module_columns.append(module_cols_dict['deep'])
        except:
            raise ValueError("the module's name is wrong")

        super().__init__(module_cols=self.module_cols,
                         init_std=init_std,
                         task=task)

        if not hidden_units:
            raise ValueError("hidden_unit can't be empty")

        #因为self.module_cols=[fm,deep],所以传参为1
        deep_input_dim = self._getInputDim(1)

        self.dnn = DNN(input_dim=deep_input_dim,
                       hidden_units=hidden_units,
                       dropout_rate=dropout_rate,
                       init_std=init_std,
                       activation=activation)
        self.outer = torch.nn.Linear(hidden_units[-1],1)

        self.fm = FM()

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        """

        :param x: tensor, with shape (batch,filed*module)
        :return:
        """
        fm_x = x[:,:len(self.module_columns[0])]
        deep_x = x[:,len(self.module_columns[0]):]
        fm_input = self._get3Dtensor(fm_x,0)  #(batch,filed,embedding_dim)
        fm = self.fm(fm_input)
        deep_input = self._get2Dtensor(deep_x,1) #(batch,filed*embedding_dim+dense)
        deep = self.outer(self.dnn(deep_input))

        return self.sigmoid(fm+deep)



