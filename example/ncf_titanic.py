# -*- coding: utf-8 -*-#
"""
Name:         ncf_titanic
Author:       路子野
Date:         2020/5/29
"""

import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
from torchctr.inputs import SparseFeat,DenseFeat
from torchctr.models.ncf import NCF
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

if __name__ == "__main__":
    df = pd.read_csv(r"E:\DataSets\titanic\train.csv")
    target = ["Survived"] ##注意，这里有个隐藏bug，以后再fix
    sparse_feat_name = ['Pclass','Sex']
    """
        缺失值处理：
            1.类别特征用'-1'填充
            2.连续特征用0填充
    """
    df[sparse_feat_name] = df[sparse_feat_name].fillna('-1', )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_feat_name:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])

    # 2.count #unique features for each sparse field,and record dense feature field name
    sparseFeats = [SparseFeat(name=name
                              ,vocabulary_size=df[name].nunique()
                              ,embedding_dim=8) for name in sparse_feat_name]

    # 3.create parameters which need by model
    module_columns_dict = {"gmf":sparseFeats,
                           "deep":sparseFeats}
    hidden_units = [256,128]

    model = NCF(module_columns_dict=module_columns_dict,
                   hidden_units=hidden_units,
                   )

    # 4.split train test dataset
    x = df[sparse_feat_name]
    y = df[target]
    trainx,testx,trainy,testy = train_test_split(x,y,test_size=0.2)
    trainx_dict,testx_dict = {},{}
    for name in sparse_feat_name:
        trainx_dict[name] = trainx[name]
        testx_dict[name] = testx[name]

    # 5.compile and train model
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["auc"])
    model.fit(trainx_dict,trainy,batch_size=32,verbose=1)
    model.test(testx_dict,testy)