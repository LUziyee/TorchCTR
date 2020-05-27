# -*- coding: utf-8 -*-#
"""
Name:         activation
Author:       路子野
Date:         2020/5/26
"""

import torch

class NonAct(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,X):
        return X


def activation_layer(act_name,hidden_size=None,dice_dim=2):
    """
    Construct activation layers
    :param act_name: str, name of activation function
    :param hidden_size: int, used of Dice activation
    :param dice_dim: int, used for Dice activation
    :return: A object of torch.nn.Module
    """
    if act_name.lower() == "sigmoid":
        act_layer = torch.nn.Sigmoid()
    elif act_name.lower() == "none":
        act_layer = NonAct()
    elif act_name.lower() == "tanh":
        act_layer = torch.nn.Tanh()
    elif act_name.lower() == "relu":
        act_layer = torch.nn.ReLU()
    else:
        raise NotImplementedError

    return act_layer


if __name__=="__main__":
    activation_layer("tanh")



