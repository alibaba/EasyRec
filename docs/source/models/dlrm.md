# DLRM

### 简介

DLRM(Deep Learning Recommendation Model for Personalization and Recommendation Systems\[Facebook\])是一种DNN模型, 支持使用连续值特征(price/age/...)和ID类特征(user_id/item_id/...), 并对特征之间的交互(interaction)进行了建模(基于内积的方式).

```
output:
                    probability of a click
model:                       |
       ____________________>DNN<______________
      /                      |                \
     /_________________>INTERACTION <__________\
    //                                         \\
  DNN                              _____________\\________
   |                              |                       |
   |                         _____|_______           _____|______
   |                        |_Emb_|____|__|    ...  |_Emb_|__|___|
input:
[ dense features ]          [sparse indices] , ..., [sparse indices]
```

### 配置说明

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
    top_dnn {
      hidden_units: [64, 32, 16]
    }

    bot_dnn {
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

- top_dnn: dense mlp的参数配置

  - hidden_units: dnn每一层的channel数目，即神经元的数目

- bot_dnn: 输出(logits)之前的mlp, 输入为dense features, sparse features and interact features.

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

### 示例Config

[DLRM_demo.config](https://easyrec.oss-cn-beijing.aliyuncs.com/config/dlrm_on_taobao.config)

### 参考论文

[DLRM](https://arxiv.org/abs/1906.00091)
