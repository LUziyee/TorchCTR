# TorchCTR

TorchCTR is a demo package of deep-learning based CTR models. You can use any high-order model with `model.fit()` and `mode.test`.

##Models List

| **Model**                      | **Paper**                                                    |
| ------------------------------ | ------------------------------------------------------------ |
| DeepFM                         | [IJCAI 2017](http://www.ijcai.org/proceedings/2017/0239.pdf) |
| Deep & Cross Network           | [ADKDD 2017](https://arxiv.org/abs/1708.05123)               |
| Neural Factorization Machine   | [SIGIR 2017](https://arxiv.org/pdf/1708.05027.pdf)           |
| Neural Collaborative Filtering | [WWW 2017](Neural Collaborative Filtering)                   |

## 设计思路

- 在点击率预估领域，通常有两种特征：类别特征(Sparse Feature)、连续特征(Dense Feature)。在TorchCTR中用两个类`SparseFeat`和`DenseFeat`来封装
- ctr的模型一般是单塔模型(e.g. FM)和双塔模型(e.g. DeepFM)等，多塔模型中每个tower在TorchCTR中被定义为组件。
- 每个组件所使用的特征可能有重合也有差异，但通常都是共享embedding layer，因此用户需要将所有使用的特征封装为feat对象，TorchCTR会为每个`SparseFeat`对象生成一个`embedding matrix`。
- 用户在实例化模型时需要传入一个字典，key为组件名，value为包含feat对象的数组
- ....

## Example



## Todo List

1. 加入Attention Layer, Sequence Layer
2. 完成AFM，DIN，DIEN
3. 完成multi-hot特征的封装与embedding api

