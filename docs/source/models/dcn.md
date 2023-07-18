# DCN

### 简介

Deep＆Cross Network（DCN）是在DNN模型的基础上，引入了一种新型的交叉网络，该网络在学习某些特征交叉时效率更高。特别是，DCN显式地在每一层应用特征交叉，不需要人工特征工程，并且只增加了很小的额外复杂性。

![dcn.png](../../images/models/dcn.png)

DCN-V2相对于前一个版本的模型，主要的改进点在于：

(1) Wide侧-Cross Network中用矩阵替代向量；

(2) 提出2种模型结构，传统的Wide&Deep并行 + Wide&Deep串行。

![dcn_v2](../../images/models/dcn_v2.jpg)
![dcn_v2_cross](../../images/models/dcn_v2_cross.jpg)

### DCN v1 配置说明

```protobuf
model_config: {
  model_class: 'DCN'
  feature_groups: {
    group_name: 'all'
    feature_names: 'user_id'
    feature_names: 'cms_segid'
    feature_names: 'cms_group_id'
    feature_names: 'age_level'
    feature_names: 'pvalue_level'
    feature_names: 'shopping_level'
    feature_names: 'occupation'
    feature_names: 'new_user_class_level'
    feature_names: 'adgroup_id'
    feature_names: 'cate_id'
    feature_names: 'campaign_id'
    feature_names: 'customer'
    feature_names: 'brand'
    feature_names: 'price'
    feature_names: 'pid'
    feature_names: 'tag_category_list'
    feature_names: 'tag_brand_list'
    wide_deep: DEEP
  }
  dcn {
    deep_tower {
      input: "all"
      dnn {
        hidden_units: [256, 128, 96, 64]
      }
    }
    cross_tower {
      input: "all"
      cross_num: 5
    }
    final_dnn {
      hidden_units: [128, 96, 64, 32, 16]
    }
    l2_regularization: 1e-6
  }
  embedding_regularization: 1e-4
}
```

- model_class: 'DCN', 不需要修改

- feature_groups: 配置一个名为'all'的feature_group。

- dcn: dcn相关的参数

- deep_tower

  - dnn: deep part的参数配置

    - hidden_units: dnn每一层的channel数目，即神经元的数目

- cross_tower

  - cross_num: 交叉层层数，默认为3

- final_dnn: 整合wide part, fm part, deep part的参数输入, 可以选择是否使用

  - hidden_units: dnn每一层的channel数目，即神经元的数目

- embedding_regularization: 对embedding部分加regularization，防止overfit

### DCN v2 配置说明

```protobuf
model_config {
  model_name: 'DCN v2'
  model_class: 'RankModel'
  feature_groups: {
    group_name: 'all'
    feature_names: 'user_id'
    feature_names: 'movie_id'
    feature_names: 'job_id'
    feature_names: 'age'
    feature_names: 'gender'
    feature_names: 'year'
    feature_names: 'genres'
    wide_deep: DEEP
  }
  backbone {
    blocks {
      name: "deep"
      inputs {
        feature_group_name: 'all'
      }
      keras_layer {
        class_name: 'MLP'
        mlp {
          hidden_units: [256, 128, 64]
        }
      }
    }
    blocks {
      name: "dcn"
      inputs {
        feature_group_name: 'all'
        input_fn: 'lambda x: [x, x]'
      }
      recurrent {
        num_steps: 3
        fixed_input_index: 0
        keras_layer {
          class_name: 'Cross'
        }
      }
    }
    concat_blocks: ['deep', 'dcn']
    top_mlp {
      hidden_units: [64, 32, 16]
    }
  }
  model_params {
    l2_regularization: 1e-4
  }
  embedding_regularization: 1e-4
}
```

- model_name: 任意自定义字符串，仅有注释作用
- model_class: 'RankModel', 不需要修改, 通过组件化方式搭建的单目标排序模型都叫这个名字
- feature_groups: 配置一个名为'all'的feature_group。
- backbone: 通过组件化的方式搭建的主干网络，[参考文档](../component/backbone.md)
  - blocks: 由多个`组件块`组成的一个有向无环图（DAG），框架负责按照DAG的拓扑排序执行个`组件块`关联的代码逻辑，构建TF Graph的一个子图
  - name/inputs: 每个`block`有一个唯一的名字（name），并且有一个或多个输入(inputs)和输出
    - input_fn: 配置一个lambda函数对输入做一些简单的变换
  - input_layer: 对输入的`feature group`配置的特征做一些额外的加工，比如执行可选的`batch normalization`、`layer normalization`、`feature dropout`等操作，并且可以指定输出的tensor的格式（2d、3d、list等）；[参考文档](../component/backbone.md#id15)
  - keras_layer: 加载由`class_name`指定的自定义或系统内置的keras layer，执行一段代码逻辑；[参考文档](../component/backbone.md#keraslayer)
  - recurrent: 循环调用指定的Keras Layer，参考 [循环组件块](../component/backbone.md#id16)
    - num_steps 配置循环执行的次数
    - fixed_input_index 配置每次执行的多路输入组成的列表中固定不变的元素
    - keras_layer: 同上
  - concat_blocks: DAG的输出节点由`concat_blocks`配置项定义，如果不配置`concat_blocks`，框架会自动拼接DAG的所有叶子节点并输出。
  - top_mlp: 各输出`组件块`的输出tensor拼接之后输入给一个可选的顶部MLP层
- model_params:
  - l2_regularization: 对DNN参数的regularization, 减少overfit
- embedding_regularization: 对embedding部分加regularization, 减少overfit

### 示例Config

1. DCN V1: [DCN_demo.config](https://easyrec.oss-cn-beijing.aliyuncs.com/config/dcn.config)
1. DCN V2: [dcn_backbone_on_movielens.config](https://github.com/alibaba/EasyRec/tree/master/examples/configs/dcn_backbone_on_movielens.config)

### 参考论文

1. [DCN v1](https://arxiv.org/abs/1708.05123)
2. [DCN v2](https://arxiv.org/abs/2008.13535)
