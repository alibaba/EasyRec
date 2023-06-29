# DLRM

### 简介

DLRM(Deep Learning Recommendation Model for Personalization and Recommendation Systems\[Facebook\])是一种DNN模型, 支持使用连续值特征(price/age/...)和ID类特征(user_id/item_id/...), 并对特征之间的交互(interaction)进行了建模(基于内积的方式).

```
output:
                    probability of a click
model:                       |
       _________________>DNN(top)<___________
      /                      |               \
     /_________________>INTERACTION <_________\
    //                                        \\
  DNN(bot)                         ____________\\_________
   |                              |                       |
   |                         _____|_______           _____|______
   |                        |_Emb_|____|__|    ...  |_Emb_|__|___|
input:
[ dense features ]          [sparse indices] , ..., [sparse indices]
```

### 配置说明

#### 1. 内置模型

```protobuf
model_config {
  model_class: 'DLRM'

  feature_groups {
    group_name: 'dense'
    feature_names: 'age_level'
    feature_names: 'pvalue_level'
    feature_names: 'shopping_level'
    feature_names: 'new_user_class_level'
    feature_names: 'price'

    wide_deep: DEEP
  }

  feature_groups {
    group_name: 'sparse'
    feature_names: 'user_id'
    feature_names: 'cms_segid'
    feature_names: 'cms_group_id'
    feature_names: 'occupation'
    feature_names: 'adgroup_id'
    feature_names: 'cate_id'
    feature_names: 'campaign_id'
    feature_names: 'customer'
    feature_names: 'brand'
    feature_names: 'pid'
    feature_names: 'tag_category_list'
    feature_names: 'tag_brand_list'

    wide_deep: DEEP
  }

  dlrm {
    bot_dnn {
      hidden_units: [64, 32, 16]
    }

    top_dnn {
      hidden_units: [128, 64]
    }

    l2_regularization: 1e-5
  }

  embedding_regularization: 1e-5
}
```

- model_class: 'DLRM', 不需要修改

- feature_groups: 特征组

  - 包含两个feature_group: dense 和sparse group, **group name不能变**

  - wide_deep: dlrm模型使用的都是Deep features, 所以都设置成DEEP

- dlrm: dlrm模型相关的参数

- bot_dnn: dense mlp的参数配置

  - hidden_units: dnn每一层的channel数目，即神经元的数目

- top_dnn: 输出(logits)之前的mlp, 输入为dense features, sparse features and interact features.

  - hidden_units: dnn每一层的channel数目，即神经元的数目

- arch_interaction_op: cat or dot

  - cat: 将dense_features和sparse features concat起来, 然后输入bot_dnn
  - dot: 将dense_features和sparse features做内积interaction, 并将interaction的结果和sparse features concat起来, 然后输入bot_dnn

- arch_interaction_itself:

  - 仅当arch_interaction_op = 'dot'时有效, features是否和自身做内积

- arch_with_dense_feature:

  - 仅当arch_interaction_op = 'dot'时有效,
    - if true, dense features也会和sparse features以及interact features concat起来, 然后进入bot_dnn.
    - 默认是false, 即仅将sparse features和interact features concat起来，输入bot_dnn.

- l2_regularization: 对DNN参数的regularization, 减少overfit

- embedding_regularization: 对embedding部分加regularization, 减少overfit

#### 2. 组件化模型

```
model_config: {
  model_name: 'DLRM'
  model_class: 'RankModel'
  feature_groups {
    group_name: 'dense'
    feature_names: 'age_level'
    feature_names: 'pvalue_level'
    feature_names: 'shopping_level'
    feature_names: 'new_user_class_level'
    feature_names: 'price'
    wide_deep: DEEP
  }
  feature_groups {
    group_name: 'sparse'
    feature_names: 'user_id'
    feature_names: 'cms_segid'
    feature_names: 'cms_group_id'
    feature_names: 'occupation'
    feature_names: 'adgroup_id'
    feature_names: 'cate_id'
    feature_names: 'campaign_id'
    feature_names: 'customer'
    feature_names: 'brand'
    feature_names: 'pid'
    feature_names: 'tag_category_list'
    feature_names: 'tag_brand_list'
    wide_deep: DEEP
  }
  backbone {
    blocks {
      name: 'bottom_mlp'
      inputs {
        feature_group_name: 'dense'
      }
      keras_layer {
        class_name: 'MLP'
        mlp {
          hidden_units: [64, 32, 16]
        }
      }
    }
    blocks {
      name: 'sparse'
      inputs {
        feature_group_name: 'sparse'
      }
      input_layer {
        output_2d_tensor_and_feature_list: true
      }
    }
    blocks {
      name: 'dot'
      inputs {
        block_name: 'bottom_mlp'
        input_fn: 'lambda x: [x]'
      }
      inputs {
        block_name: 'sparse'
        input_fn: 'lambda x: x[1]'
      }
      keras_layer {
        class_name: 'DotInteraction'
      }
    }
    blocks {
      name: 'sparse_2d'
      inputs {
        block_name: 'sparse'
        input_fn: 'lambda x: x[0]'
      }
    }
    concat_blocks: ['bottom_mlp', 'sparse_2d', 'dot']
    top_mlp {
      hidden_units: [256, 128, 64]
    }
  }
  model_params {
    l2_regularization: 1e-5
  }
  embedding_regularization: 1e-5
}
```

- model_name: 任意自定义字符串，仅有注释作用

- model_class: 'RankModel', 不需要修改, 通过组件化方式搭建的单目标排序模型都叫这个名字

- feature_groups: 特征组

  - 包含两个feature_group: dense 和sparse group

  - wide_deep: dlrm模型使用的都是Deep features, 所以都设置成DEEP

- backbone: 通过组件化的方式搭建的主干网络，[参考文档](../component/backbone.md)

  - blocks: 由多个`组件块`组成的一个有向无环图（DAG），框架负责按照DAG的拓扑排序执行个`组件块`关联的代码逻辑，构建TF Graph的一个子图
  - name/inputs: 每个`block`有一个唯一的名字（name），并且有一个或多个输入(inputs)和输出
  - input_layer: 对输入的`feature group`配置的特征做一些额外的加工，比如执行可选的`batch normalization`、`layer normalization`、`feature dropout`等操作，并且可以指定输出的tensor的格式（2d、3d、list等）；[参考文档](../component/backbone.md#id15)
  - keras_layer: 加载由`class_name`指定的自定义或系统内置的keras layer，执行一段代码逻辑；[参考文档](../component/backbone.md#keraslayer)
  - concat_blocks: DAG的输出节点由`concat_blocks`配置项定义
  - top_mlp: 各输出`组件块`的输出tensor拼接之后输入给一个可选的顶部MLP层

- model_params:

  - l2_regularization: 对DNN参数的regularization, 减少overfit

- embedding_regularization: 对embedding部分加regularization, 减少overfit

### 示例Config

1. 内置模型：[DLRM_demo.config](https://easyrec.oss-cn-beijing.aliyuncs.com/config/dlrm_on_taobao.config)
1. 组件化模型：[dlrm_backbone_on_criteo.config](https://github.com/alibaba/EasyRec/tree/master/examples/configs/dlrm_backbone_on_criteo.config)

### 参考论文

[DLRM](https://arxiv.org/abs/1906.00091)
