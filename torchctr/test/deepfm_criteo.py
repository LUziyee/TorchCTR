# -*- coding: utf-8 -*-#
"""
Name:         deepfm_criteo
Author:       路子野
Date:         2020/5/27
"""

import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
from torchctr.inputs import SparseFeat,DenseFeat
from torchctr.models.deepfm import DeepFM
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

if __name__ == "__main__":
    df = pd.read_csv("D:\MyPrograme\Python\TorchRec\参考\DeepCTR-Torch\examples\criteo_sample.txt")
    label = ["label"]
    sparse_feat_name = ["C"+str(i) for i in range(1,27)]
    dense_feat_name = ["I"+str(i) for i in range(1,14)]
    """
        缺失值处理：
            1.类别特征用'-1'填充
            2.连续特征用0填充
    """
    df[sparse_feat_name] = df[sparse_feat_name].fillna('-1', )
    df[dense_feat_name] = df[dense_feat_name].fillna(0, )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_feat_name:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_feat_name] = mms.fit_transform(df[dense_feat_name])

    # 2.count #unique features for each sparse field,and record dense feature field name
    target = ['label']
    sparseFeats = [SparseFeat(name=name
                              ,vocabulary_size=df[name].nunique()
                              ,embedding_dim=8) for name in sparse_feat_name]
    denseFeats = [DenseFeat(name=name) for name in dense_feat_name]

    # 3.create parameters which need by model
    module_columns_dict = {"fm":sparseFeats,
                           "deep":sparseFeats+denseFeats}
    hidden_units = [256,128,64]

    model = DeepFM(module_columns_dict=module_columns_dict,
                   hidden_unit=hidden_units,
                   )

    # 4.split train test dataset
    x = df[sparse_feat_name+dense_feat_name]
    y = df[target]
    trainx,testx,trainy,testy = train_test_split(x,y,test_size=0.2,random_state=2020,shuffle=True)
    trainx_dict,testx_dict = {},{}
    for name in sparse_feat_name+dense_feat_name:
        trainx_dict[name] = trainx[name]
        testx_dict[name] = testx[name]

    # 5.compile and train model
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["auc"])
    model.fit(trainx_dict,trainy)
    for name,tensor in model.named_parameters():
        print(name)
