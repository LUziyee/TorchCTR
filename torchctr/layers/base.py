# -*- coding: utf-8 -*-#
"""
Name:         base
Author:       路子野
Date:         2020/5/26
"""

import torch
from torchctr.layers.activation import activation_layer

class DNN(torch.nn.Module):
    '''
    Input shape:
        - nD tensor with shape: ``(batch_size, ..., input_dim)``
        The most common situation would be a 2D input with shape ``(batch_size, input_dim)``
        When meet sequence task, the input will be a 3D tensor.

    Output shape:
        - nD tensor with shape:``(batch_size, ..., hidden_size[-1])```.
        For example, for a 2D input with shape ``(batch_size,input_dim)``, the output would have shape ``(batch_size,hidden_size[-1])
    '''

    def __init__(self,input_dim,hidden_units,activation='relu',dropout_rate=0.5,init_std=0.001):
        """
        initialize dnn layer
        :param input_dim: Integer input feature dimension.
        :param hidden_units: List list of positive integer, record the num of units in each hidden layer.
        :param activation: String activation function to use in each hidden layer's output
        :param dropout_rate: float used to initialize layers' weight
        :param init_std:
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        if not self.hidden_units:
            raise ValueError("hidden_units is empty!! Can't generate a dnn object.")
        hidden_units = [input_dim] + hidden_units

        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_units[i],hidden_units[i+1]) for i in range(len(hidden_units)-1)]
        )

        self.activation_layers = torch.nn.ModuleList(
            [activation_layer(self.activation) for i in range(len(hidden_units)-1)]
        )

        for name,tensor in self.linears.named_parameters():
            if "weight" in name:
                torch.nn.init.normal_(tensor,mean=0,std=init_std)

    def forward(self,inputs):
        """
        DNN forward propagation
        :param inputs: nD tensor with shape: ``(batch_size, ..., input_dim)``
        :return: nD tensor with shape:``(batch_size, ..., hidden_size[-1])```
        """
        deep_input = inputs

        for i in range(len(self.linears)):
            tmp = self.linears[i](deep_input)
            tmp = self.activation_layers[i](tmp)
            deep_input = self.dropout(tmp)
        return deep_input


class PredictionLayer(torch.nn.Module):
    """
    Used to be the output layer in model, the different task will return different layer
    The PredictionLayer's input is the output of model's last layer.
    In fact, it just to use different activation function.
    For example, for the binary classification task, should use sigmoid activation function to deal input.
    """
    def __init__(self,task="binary",use_bias=True):
        """

        :param task: String name of task
        :param use_bias: bool
        """
        super().__init__()

        self.task = task
        self.use_bias = use_bias
        if self.task == "binary":
            self.act_layer = activation_layer("sigmoid")
        elif self.task == "regression":
            self.act_layer = activation_layer("none")
        elif self.task == "multiclass":
            self.act_layer = activation_layer("none")
        else:
            raise ValueError("task must be binary,multiclass or regression")

        if self.use_bias:
            self.bais = torch.nn.Parameter(torch.zeros((1,)))

    def forward(self,inputs):
        """

        :param inputs: the layer's output of model
        :return: final output of model
        """
        if self.use_bias:
            inputs += self.bais
        return self.act_layer(inputs)



























