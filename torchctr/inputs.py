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
    def __init__(self,name,vocabulary_size,embedding_dim):
        """

        :param name: String sparse feature name
        :param vocabulary_size: Integer
        :param embedding_dim: Integer
        """
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim


class DenseFeat():
    """

    """
    def __init__(self,name,):
        """

        :param name: String dense feature nanme
        """
        self.name = name
        self.dim = 1


def creatEmbeddingMatrix(feat_columns,init_std):
    """
    generate embedding matrix object for sparse feature
    :param feat_columns: List include SparseFeat objects and DenseFeat Objects
    :param init_std: float
    :return:
    """
    embedding_dict = torch.nn.ModuleDict()
    for feat in feat_columns:
        if isinstance(feat,SparseFeat):
            embedding_dict[feat.name] = torch.nn.Embedding(feat.vocabulary_size,feat.embedding_dim)

    for matrix in embedding_dict.values():
        torch.nn.init.normal_(matrix.weight,mean=0,std=init_std)
    return embedding_dict















