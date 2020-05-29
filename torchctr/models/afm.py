# -*- coding: utf-8 -*-#
"""
Name:         afm
Author:       路子野
Date:         2020/5/29
"""

import torch
import torch.nn as nn
from torchctr.inputs import SparseFeat,DenseFeat
from torchctr.models.basemodel import BaseModel
from torchctr.layers.interaction import AttentionBasedPooling
import warnings
warnings.filterwarnings("ignore")

class AFM(BaseModel):
    """
    Instantiates the Attentional FM architecture.
    """
    def __init__(self,module_columns_dict,init_std=0.0001,
                 task="binary",hidden_units=[128,64],dropout_rate=0.5,
                 activation="relu",init_method="normal"):
        module_columns = []
        try:
            module_columns.append(module_columns_dict["afm"])
        except:
            raise ValueError("the module's name is wrong")

        super().__init__(module_columns=module_columns,
                         init_std=init_std,
                         task=task)

        self.afm = AttentionBasedPooling(embedding_dim=self.module_columns[0][0].embedding_dim,
                                         hidden_units=hidden_units,
                                         activation=activation,
                                         dropout_rate=dropout_rate,
                                         init_std=init_std)  #todo list：init_method
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        afm_input = self._get3Dtensor(x,0)
        afm = self.sigmoid(self.afm(afm_input))
        return afm
