# MultiTower

### 简介

- 多塔模型效果比单塔模型有明显的提升
- 不采用FM，所以embedding可以有不同的dimension。

![multi_tower.png](../../images/models/multi_tower.png)

### 模型配置

#### 1. 内置模型

```protobuf
model_config: {
  model_class: 'MultiTower'
  feature_groups: {
    group_name: 'user'
    feature_names: 'user_id'
    feature_names: 'cms_segid'
    ...
    feature_names: 'new_user_class_level'
    wide_deep: DEEP
  }
  feature_groups: {
    group_name: 'item'
    feature_names: 'adgroup_id'
    feature_names: 'cate_id'
    ...
    feature_names: 'price'
    wide_deep: DEEP
  }
  feature_groups: {
    group_name: 'combo'
    feature_names: 'pid'
    feature_names: 'tag_category_list'
    feature_names: 'tag_brand_list'
    wide_deep: DEEP
  }
  losses {
    loss_type: F1_REWEIGHTED_LOSS
    weight: 1.0
    f1_reweighted_loss {
      f1_beta_square: 1.0
    }
  }
  losses {
    loss_type: PAIR_WISE_LOSS
    weight: 1.0
  }
  multi_tower {
    towers {
      input: "user"
      dnn {
        hidden_units: [256, 128, 96, 64]
      }
    }
    towers {
      input: "item"
      dnn {
        hidden_units: [256, 128, 96, 64]
      }
    }
    towers {
      input: "combo"
      dnn {
        hidden_units: [128, 96, 64, 32]
      }
    }
    final_dnn {
      hidden_units: [128, 96, 64, 32, 16]
    }
    l2_regularization: 1e-6
  }
  embedding_regularization: 1e-4
}
```

- feature_groups: 不同的特征组，如user feature为一组，item feature为一组, combo feature为一组
  - group_name: 可以根据实际情况取
  - wide_deep: 必须是DEEP
- losses: 可选，可以选择同时配置两个loss函数，并且为每个loss配置不同的权重
  - loss_type: CLASSIFICATION [默认值] 二分类的sigmoid cross entropy loss
  - loss_type: PAIR_WISE_LOSS [可选] 以优化AUC为主要目标的 pairwise rank loss
  - loss_type: F1_REWEIGHTED_LOSS [可选] 可以调节二分类模型recall/precision相对权重的loss; 注意不要与`loss_type: CLASSIFICATION`同时使用
- f1_reweight_loss: 可以调节二分类模型`recall/precision`相对权重的损失函数
  - f1_beta_square: 大于1的值会导致模型更关注`recall`，小于1的值会导致模型更关注`precision`
  - F1 分数，又称平衡F分数（balanced F Score），它被定义为精确率和召回率的调和平均数。
    - ![](../../images/other/f1_score.svg)
  - 更一般的，我们定义 `F_beta` 分数为:
    - ![](../../images/other/f_beta_score.svg)
  - f1_beta_square 即为 上述公式中的 beta 系数的平方。
- towers:
  - 每个feature_group对应了一个tower, tower的input必须和feature_groups的group_name对应
  - dnn: 深度网络
    - hidden_units: 定义不同层的channel数目，即神经元数目
- final_dnn 整合towers和din_towers的输入
  - hidden_units: dnn每一层的channel数目，即神经元的数目
- l2_regularization: L2正则，防止overfit
- embedding_regularization: embedding的L2正则

#### 2. 组件化模型

```protobuf
model_config: {
  model_name: 'MultiTower'
  model_class: 'RankModel'
  feature_groups: {
    group_name: 'user'
    feature_names: 'user_id'
    feature_names: 'cms_segid'
    feature_names: 'cms_group_id'
    feature_names: 'age_level'
    feature_names: 'pvalue_level'
    feature_names: 'shopping_level'
    feature_names: 'occupation'
    feature_names: 'new_user_class_level'
    wide_deep: DEEP
  }
  feature_groups: {
    group_name: 'item'
    feature_names: 'adgroup_id'
    feature_names: 'cate_id'
    feature_names: 'campaign_id'
    feature_names: 'customer'
    feature_names: 'brand'
    feature_names: 'price'
    wide_deep: DEEP
  }
  feature_groups: {
    group_name: 'combo'
    feature_names: 'pid'
    feature_names: 'tag_category_list'
    feature_names: 'tag_brand_list'
    wide_deep: DEEP
  }
  losses {
    loss_type: F1_REWEIGHTED_LOSS
    weight: 1.0
    f1_reweighted_loss {
      f1_beta_square: 2.25
    }
  }
  losses {
    loss_type: PAIR_WISE_LOSS
    weight: 1.0
  }
  backbone {
    blocks {
      name: "user_tower"
      inputs {
        feature_group_name: "user"
      }
      keras_layer {
        class_name: "MLP"
        mlp {
          hidden_units: [256, 128]
        }
      }
    }
    blocks {
      name: "item_tower"
      inputs {
        feature_group_name: "item"
      }
      keras_layer {
        class_name: "MLP"
        mlp {
          hidden_units: [256, 128]
        }
      }
    }
    blocks {
      name: "combo_tower"
      inputs {
        feature_group_name: "combo"
      }
      keras_layer {
        class_name: "MLP"
        mlp {
          hidden_units: [256, 128]
        }
      }
    }
    blocks {
      name: "top_mlp"
      inputs {
        block_name: "user_tower"
      }
      inputs {
        block_name: "item_tower"
      }
      inputs {
        block_name: "combo_tower"
      }
      keras_layer {
        class_name: "MLP"
        mlp {
          hidden_units: [256, 128, 64]
        }
      }
    }
  }
  model_params {
    l2_regularization: 1e-6
  }
  embedding_regularization: 1e-4
}
```

- model_name: 任意自定义字符串，仅有注释作用
- model_class: 'RankModel', 不需要修改, 通过组件化方式搭建的单目标排序模型都叫这个名字
- feature_groups: 特征组
  - 可包含多个feature_group: 如 user、item、combo
  - wide_deep: multi_tower模型使用的都是Deep features, 所以都设置成DEEP
- backbone: 通过组件化的方式搭建的主干网络，[参考文档](../component/backbone.md)
  - blocks: 由多个`组件块`组成的一个有向无环图（DAG），框架负责按照DAG的拓扑排序执行个`组件块`关联的代码逻辑，构建TF Graph的一个子图
  - name/inputs: 每个`block`有一个唯一的名字（name），并且有一个或多个输入(inputs)和输出
  - keras_layer: 加载由`class_name`指定的自定义或系统内置的keras layer，执行一段代码逻辑；[参考文档](../component/backbone.md#keraslayer)
  - concat_blocks: DAG的输出节点由`concat_blocks`配置项定义，如果不配置`concat_blocks`，框架会自动拼接DAG的所有叶子节点并输出。
- model_params:
  - l2_regularization: 对DNN参数的regularization, 减少overfit
- embedding_regularization: 对embedding部分加regularization, 减少overfit

### 示例config

1. 内置模型：[multi_tower_demo.config](https://easyrec.oss-cn-beijing.aliyuncs.com/config/multi-tower.config)
1. 组件化模型：[multi_tower_backbone_on_taobao.config](https://github.com/alibaba/EasyRec/tree/master/samples/model_config/multi_tower_backbone_on_taobao.config)

### 参考论文

自研模型，暂无参考论文
