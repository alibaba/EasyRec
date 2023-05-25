# WideAndDeep

### 简介

WideAndDeep包含Wide和Deep两部分，Wide部分负责记忆，Deep部分负责泛化。Wide部分可以做显式的特征交叉，Deep部分可以实现隐式自动的特征交叉。

![wide_and_deep.png](../../images/models/wide_and_deep.png)

### 参考论文

[WideAndDeep](https://arxiv.org/abs/1606.07792)

### 配置说明

```protobuf
model_config:{
  model_class: "WideAndDeep"
  feature_groups: {
    group_name: "deep"
    feature_names: "user_id"
    feature_names: "movie_id"
    ...
    wide_deep:DEEP
  }
  feature_groups: {
    group_name: "wide"
    feature_names: "user_id"
    feature_names: "movie_id"
    ...
    wide_deep:WIDE
  }

  wide_and_deep {
    wide_output_dim: 16
    dnn {
      hidden_units: [256, 128, 64]
    }

    final_dnn {
      hidden_units: [64, 32, 16]
    }
    l2_regularization: 1e-4
  }
  embedding_regularization: 1e-4
}
```

- model_class: 'WideAndDeep', 不需要修改

- feature_groups:

  需要两个feature_group: wide group和deep group, **group name不能变**

- wide_and_deep:  wide_and_deep 相关的参数

- dnn: deep part的参数配置

  - hidden_units: dnn每一层的channel数目，即神经元的数目

- wide_output_dim: wide部分输出的大小

- final_dnn: 整合wide part, deep part的参数输入, 可以选择是否使用

  - hidden_units: dnn每一层的channel数目，即神经元的数目

- embedding_regularization: 对embedding部分加regularization，防止overfit

- input_type: 如果在提交到pai-tf集群上面运行，读取 MaxCompute 表作为输入数据，data_config：input_type要设置为OdpsInputV2。

### 示例Config

[wide_and_deep_on_movielens.config](../configs/wide_and_deep_on_movieslen.config)
