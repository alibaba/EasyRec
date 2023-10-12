# Highway Network

### 简介

传统的神经网络随着深度的增加，训练越来越困难。Highway Network使用简单的SGD就可以训练很深的网络，收敛速度更快。并且Highway Network还可以用来以增量的方式微调预训练好的embedding特征。

### 配置说明

```protobuf
model_config: {
  model_name: 'HighWayNetwork'
  model_class: 'RankModel'
  feature_groups: {
    group_name: 'image'
    feature_names: 'embedding'
    wide_deep: DEEP
  }
  feature_groups: {
    group_name: 'general'
    feature_names: 'user_id'
    feature_names: 'movie_id'
    feature_names: 'gender'
    feature_names: 'age'
    feature_names: 'occupation'
    feature_names: 'zip_id'
    feature_names: 'movie_year_bin'
    feature_names: 'title'
    feature_names: 'genres'
    feature_names: 'score_year_diff'
    feature_names: 'score_time'
    wide_deep: DEEP
  }
  backbone {
    blocks {
      name: 'highway'
      inputs {
        feature_group_name: 'image'
      }
      keras_layer {
        class_name: 'Highway'
      }
    }
    blocks {
      name: 'top_mlp'
      inputs {
        feature_group_name: 'general'
      }
      inputs {
        block_name: 'highway'
      }
      keras_layer {
        class_name: 'MLP'
        mlp {
          hidden_units: [256, 128, 64]
        }
      }
    }
  }
  model_params {
    l2_regularization: 1e-6
  }
  embedding_regularization: 1e-6
}
```

- model_name: 任意自定义字符串，仅有注释作用
- model_class: 'RankModel', 不需要修改, 通过组件化方式搭建的单目标排序模型都叫这个名字
- feature_groups: 配置一组特征。
- backbone: 通过组件化的方式搭建的主干网络，[参考文档](../component/backbone.md)
  - blocks: 由多个`组件块`组成的一个有向无环图（DAG），框架负责按照DAG的拓扑排序执行个`组件块`关联的代码逻辑，构建TF Graph的一个子图
  - name/inputs: 每个`block`有一个唯一的名字（name），并且有一个或多个输入(inputs)和输出
  - keras_layer: 加载由`class_name`指定的自定义或系统内置的keras layer，执行一段代码逻辑；[参考文档](../component/backbone.md#keraslayer)
  - Highway: 使用Highway Network微调图像embedding。组件的参数，详见[参考文档](../component/component.md#id2)
  - concat_blocks: DAG的输出节点由`concat_blocks`配置项定义，如果不配置`concat_blocks`，框架会自动拼接DAG的所有叶子节点并输出。
- model_params:
  - l2_regularization: (可选) 对DNN参数的regularization, 减少overfit
- embedding_regularization: 对embedding部分加regularization, 减少overfit

### 示例Config

[highway_on_movielens.config](https://github.com/alibaba/EasyRec/tree/master/samples/model_config/highway_on_movielens.config)

### 参考论文

[Highway Network](https://arxiv.org/pdf/1505.00387.pdf)
