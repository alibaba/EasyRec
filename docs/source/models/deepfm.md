# DeepFM

### 简介

DeepFM是在WideAndDeep基础上加入了FM模块的改进模型。FM模块和DNN模块共享相同的特征，即相同的Embedding。
注意：经过我们的扩展，DeepFM支持不同特征使用不同大小的embedding size。

![deepfm.png](../../images/models/deepfm.png)

### 配置说明

```protobuf
model_config:{
  model_class: "DeepFM"
  feature_groups: {
    group_name: "deep"
    feature_names: "hour"
    feature_names: "c1"
    ...
    feature_names: "site_id_app_id"
    wide_deep:DEEP
  }
  feature_groups: {
    group_name: "wide"
    feature_names: "hour"
    feature_names: "c1"
    ...
    feature_names: "c21"
    wide_deep:WIDE
  }

  deepfm {
    wide_output_dim: 16

    dnn {
      hidden_units: [128, 64, 32]
    }

    final_dnn {
      hidden_units: [128, 64]
    }
    l2_regularization: 1e-5
  }
  embedding_regularization: 1e-7
}
```

- model\_class: 'DeepFM', 不需要修改

- feature\_groups:

  需要两个feature\_group: wide group和deep group, **group name不能变**

- deepfm:  deepfm相关的参数

- dnn: deep part的参数配置

  - hidden\_units: dnn每一层的channel数目，即神经元的数目

- wide\_output\_dim: wide部分输出的大小

- final\_dnn: 整合wide part, fm part, deep part的参数输入, 可以选择是否使用

  - hidden\_units: dnn每一层的channel数目，即神经元的数目

- embedding\_regularization: 对embedding部分加regularization，防止overfit

- input\_type: 如果在提交到pai-tf集群上面运行，读取max compute 表作为输入数据，data\_config：input\_type要设置为OdpsInputV2。

### 示例Config

[DeepFM\_demo.config](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/config/deepfm.config)

### 参考论文

[DeepFM](https://arxiv.org/abs/1703.04247)
