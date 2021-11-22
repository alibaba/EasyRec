# DropoutNet

### 简介

类似于DSSM的双塔召回模型，分为user塔和item塔。DropoutNet是一种既适用于头部用户和物品，也适用于中长尾的、甚至全新的用户和物品的召回模型。

原始的DropoutNet需要提供用户和物品的embedding向量作为输入监督信号，使得模型使用门槛增高。

EasyRec的实现对原始DropoutNet模型进行了改造，直接使用用户与物品的交互行为数据作为训练目标进行端到端训练，从而避免了需要使用其他模型提供用户和物品的embedding作为监督信号。
相应地，我们对模型的损失函数也进行了改造，如下图所示。

EasyRec的实现使用了Negative Mining的负采样技术，在训练过程中从当前mini batch中采样负样本，扩大了样本空间，使得学习更加高效，同时适用于训练数据量比较少的场景。

![dropoutnet](../../images/models/dropoutnet.png)

### 配置说明

```protobuf
model_config {
  model_class: "DropoutNet"
  feature_groups: {
    group_name: 'user_content'
    feature_names: 'user_id'
    feature_names: 'gender'
    ...
    feature_names: 'city'
    wide_deep:DEEP
  }
  feature_groups: {
    group_name: 'user_preference'
    feature_names: 'fans_num'
    feature_names: 'follow_num'
    ...
    feature_names: 'click_cnt_7d'
    wide_deep:DEEP
  }
  feature_groups: {
    group_name: "item_content"
    feature_names: 'is_new'
    feature_names: 'primary_type'
    ...
    feature_names: 'grade_score'
    wide_deep:DEEP
  }
  feature_groups: {
    group_name: "item_preference"
    feature_names: 'exposure_cnt_7d'
    feature_names: 'click_cnt_7d'
    ...
    feature_names: 'share_cnt_15d'
    wide_deep:DEEP
  }
  dropoutnet {
    user_content {
      hidden_units: [256]
    }
    item_content {
      hidden_units: [256]
    }
    user_preference {
      hidden_units: [512]
    }
    item_preference {
      hidden_units: [512]
    }
    user_tower {
      hidden_units: [256, 128]
    }
    item_tower {
      hidden_units: [256, 128]
    }
    l2_regularization: 1e-06
  }
  embedding_regularization: 5e-5
}
```

- model_class: 'DropoutNet', 不需要修改
- feature_groups: 需要四个feature_group: user_content、user_preference和item_content、item_preference, **group name不能变**。
  其中，user_content和user_preference两者至少要有1个；item_content和item_preference两者至少要有1个。
- dropoutnet: dropoutnet相关的参数，必须配置user_tower和item_tower
- user_content/user_content/user_preference/item_preference/user_tower/item_tower: dnn的参数配置
    - hidden_units: dnn每一层的channel数目，即神经元的数目
- embedding_regularization: 对embedding部分加regularization，防止overfit

### 示例Config

[DropoutNet_demo.config](https://easyrec.oss-cn-beijing.aliyuncs.com/config/dropoutnet.config)

### 效果评估

[效果评估](https://easyrec.oss-cn-beijing.aliyuncs.com/docs/recall_eval.pdf)

### 参考论文

[DropoutNet.pdf](https://papers.nips.cc/paper/2017/file/dbd22ba3bd0df8f385bdac3e9f8be207-Paper.pdf)
