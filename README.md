# Update
弃坑...

# TorchCTR

TorchCTR is a demo package of deep-learning based CTR models. You can use any high-order model with `model.fit()` and `model.test()`.

## Models List

| **Model**                          | **Paper**                                                    |
| ---------------------------------- | ------------------------------------------------------------ |
| DeepFM                             | [IJCAI 2017](http://www.ijcai.org/proceedings/2017/0239.pdf) |
| Deep & Cross Network               | [ADKDD 2017](https://arxiv.org/abs/1708.05123)               |
| Neural Factorization Machine       | [SIGIR 2017](https://arxiv.org/pdf/1708.05027.pdf)           |
| Neural Collaborative Filtering     | [WWW 2017](http://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf) |
| Attentional Factorization Machines | [IJCAI 2017](https://arxiv.org/abs/1708.04617)               |

## 设计思路

- 在点击率预估领域，通常有两种特征：类别特征(Sparse Feature)、连续特征(Dense Feature)。在TorchCTR中用两个类`SparseFeat`和`DenseFeat`来封装
- ctr的模型一般是单塔模型(e.g. FM)和双塔模型(e.g. DeepFM)等，多塔模型中每个tower在TorchCTR中被定义为组件。
- 每个组件所使用的特征可能有重合也有差异，但通常都是共享embedding layer，因此用户需要将所有使用的特征封装为feat对象，TorchCTR会为每个`SparseFeat`对象生成一个`embedding matrix`。
- 用户在实例化模型时需要传入一个字典，key为组件名，value为包含feat对象的数组，TorchCTR会记录下该模型每个组件下的每个特征及特征的排列顺序。
- 用户在编译模型时需要指定模型的损失函数，评价指标，学习率等。
- 模型接收的数据也是一个字典，key为特征名，value为包含该特征所有数据的数组，用户在传入之前应做好缺失值处理以及自然数编码(for sparse feature)或归一化(for dense feature)。TorchCTR接收数据后会自动按实例化模型时传入的各个组件的特征来生成模型真实的input。

## Todo List

1. 加入Attention Layer, Sequence Layer
2. 完成DIN，DIEN，PNN
3. 完成multi-hot特征的封装与embedding api
4. 完成cuda的支持
5. 有些接口还有些混乱和冗余，需要规范

