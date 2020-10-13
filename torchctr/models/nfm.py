# -*- coding: utf-8 -*-#
"""
Name:         NFM
Author:       路子野
Date:         2020/5/28
"""
from torchctr.inputs import SparseFeat,DenseFeat
from torchctr.models.basemodel import BaseModel
from torchctr.layers.base import DNN
from torchctr.layers.interaction import BiInteractionPooling
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class NFM(BaseModel):
    """
    Instantiates the NFM Network architecture.
    """
    def __init__(self,module_columns_dict,init_std=0.0001,
                 task="binary",hidden_units=[128,64],dropout_rate=0.5,
                 activation="relu",init_method="normal"):

        module_columns = []
        try:
            module_columns.append(module_columns_dict["base"])
        except:
            raise ValueError("the module's name is wrong")

        super().__init__(module_columns=module_columns,
                         init_std=init_std,
                         task=task)
        self.biinteractionpooling = BiInteractionPooling()
        input_dim = module_columns[0][0].embedding_dim
        self.dnn = DNN(input_dim=input_dim,
                       hidden_units=hidden_units,
                       dropout_rate=dropout_rate,
                       init_std=init_std,
                       activation=activation)

        self.outer = torch.nn.Linear(hidden_units[-1],1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        """
        :param x: A 3D tensor with shape:(batch,field)
        :return: 2D tensor with shape:(batch,1)
        """
        embedding_list = []
        for index, feat in enumerate(self.module_cols[0]):
            if isinstance(feat, SparseFeat):
                feat_id = x[:, [index]].long()
                embedding_list.append(self.embed_dict[feat.name](feat_id)) #[(batch,1,embedding_dim)]
            else:
                raise ValueError("nfm model can not have dense feature!")
        pool_input = torch.cat(embedding_list, dim=1)  # (batch,filed,embedding_dim)
        pool = self.biinteractionpooling(pool_input)
        deep = self.dnn(pool)
        return self.sigmoid(self.outer(deep))


