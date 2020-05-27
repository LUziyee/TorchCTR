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
    def __init__(self, module_columns_dict, hidden_unit=[256,128], task="binary",
                 init_std=0.0001, learning_rate=0.001,dropout_rate=0,):
        """

        :param module_columns_dict: dict, {feat_name:[sparsefeat1,sparsefeat2,densefeat1,...]}
        :param hidden_unit:list, default=[256,128,64]
        :param task: string,
        :param init_std: float, used to initialize layer weight and embedding weight
        :param learning_rate: float,
        """
        self.module_columns = []  #存储所有组件的 特征对象 ，tips:特征对象是有重复的
        try:
            #把各个组件的特征对象放进list去
            self.module_columns.append(module_columns_dict["fm"])
            self.module_columns.append(module_columns_dict['deep'])
        except:
            raise ValueError("this is not include fm module's feature")

        super().__init__(module_columns=self.module_columns,
                         init_std=init_std,
                         task=task,
                         learning_rate=learning_rate)

        if not hidden_unit:
            raise ValueError("hidden_unit can't be empty")

        deep_input_dim = self._getInputDim()

        self.dnn = DNN(input_dim=deep_input_dim,hidden_units=hidden_unit,dropout_rate=dropout_rate,init_std=init_std)
        self.outer = torch.nn.Linear(hidden_unit[-1],1)

        self.fm = FM()

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        fm_x = x[:,:len(self.module_columns[0])]
        deep_x = x[:,len(self.module_columns[0]):]
        fm_embedding_list = []  #final (filed,batch,1,embedding_dim)
        for index,feat in enumerate(self.module_columns[0]):
            feat_id = fm_x[:,[index]].long()  #(batch,1)
            fm_embedding_list.append(self.embedding_dict[feat.name](feat_id))
        fm_input = torch.cat(fm_embedding_list,dim=1) #(batch,filed,embedding_dim)
        fm = self.fm(fm_input)
        deep_embedding_list = []
        deep_dense_list = []
        for index,feat in enumerate(self.module_columns[1]):
            if isinstance(feat,SparseFeat):
                feat_id = deep_x[:,[index]].long()
                deep_embedding_list.append(self.embedding_dict[feat.name](feat_id).squeeze(dim=1))
            else:
                deep_dense_list.append(deep_x[:,[index]].float())
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
