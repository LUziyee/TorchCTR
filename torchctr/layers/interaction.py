# -*- coding: utf-8 -*-#
#-------------------------------------------------------------------------------
# Name:         interaction
# Description:
# Author:       路子野
# Date:         2020/5/26
#-------------------------------------------------------------------------------

import torch
from .activation import activation_layer

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
