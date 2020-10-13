# -*- coding: utf-8 -*-#
"""
# Name:         inputs
# Description:
# Author:       路子野
# Date:         2020/5/27
"""

import torch
import torch.nn as nn

class SparseFeat():
    """

    """
    def __init__(self,name,vocab_size,embed_dim):
        """

        :param name: String, means sparse feature's name
        :param vocab_size: Integer, means vocabulary size
        :param embed_dim: Integer, means embedding vector's length
        """
        self.name = name
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim


class DenseFeat():
    """

    """
    def __init__(self,name,dim=1):
        """

        :param name: String, means dense feature's nanme
        """
        self.name = name
        self.dim = dim
















