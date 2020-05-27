# -*- coding:utf-8 -*-
"""
@Time:19:03
@Author:LuZiye
"""

import torch
from .basemodel import BaseModel
from ..layers.base import DNN
from ..inputs import SparseFeat,DenseFeat
from ..layers.interaction import FM


class DeepFM(BaseModel):
    def __init__(self, module_columns_dict, hidden_unit, task="binary", init_std=0.001, learning_rate=0.001,dropout_rate=0.5,):
        """

        :param module_columns_dict:
        :param hidden_unit:
        :param task:
        :param init_std:
        :param learning_rate:
        """
        self.module_columns = []
        try:
            self.module_columns.append(module_columns_dict["fm"])
            self.module_columns.append(module_columns_dict['deep'])
        except:
            raise ValueError("this is not include fm module's feature")

        super().__init__(self.module_columns, task, init_std, learning_rate)

        if not hidden_unit:
            raise ValueError("hidden_unit can't be empty")

        deep_input_dim = self._getInputDim()

        self.dnn = DNN(input_dim=deep_input_dim,hidden_units=hidden_unit,)
        self.outer = torch.nn.Linear(hidden_unit[-1],1)

        self.fm = FM()

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        fm_x = x[:,len(self.module_columns[0])]
        deep_x = x[:,len(self.module_columns[0]):]
        fm_embedding_list = []  #final (filed,batch,1,embedding_dim)
        for index,feat in enumerate(self.module_columns[0]):
            feat_id = fm_x[:,[index]]  #(batch,1)
            fm_embedding_list.append(self.embedding_dict[feat.name](feat_id))
        fm_input = torch.cat(fm_embedding_list,dim=1) #(batch,filed,embedding_dim)
        fm = self.fm(fm_input)
        deep_embedding_list = []
        deep_dense_list = []
        for index,feat in enumerate(self.module_columns[1]):
            if isinstance(feat,SparseFeat):
                feat_id = deep_x[:,[index]]
                deep_embedding_list.append(self.embedding_dict[feat.name](feat_id).squeeze(dim=1))
            else:
                deep_dense_list.append(deep_x[:[index]])
        deep_input = torch.cat(deep_embedding_list+deep_dense_list,dim=1)  #(batch,filed*embedding_dim+dense)
        deep = self.outer(self.dnn(deep_input))

        return self.sigmoid(fm+deep)


    def _getInputDim(self):
        deep_module = self.module_columns[1]
        input_dim = 0
        for feat in deep_module:
            if isinstance(feat,SparseFeat):
                input_dim += feat.embedding_dim
            else:
                input_dim += feat.dim
        return input_dim
