# ESMM

### 简介

CVR预估模型的本质，不是预测“item被点击，然后被转化”的概率（CTCVR），而是“假设item被点击，那么它被转化”的概率（CVR）。\\它与CTR没有绝对的关系，很多人有一个先入为主的认知，即若user对某item的点击概率很低，则user对这个item的转化概率也肯定低，这是不成立的。
这就是不能直接使用全部样本训练CVR模型的原因，因为不知道那些unclicked的item，假设他们被点击了，是否会被转化。如果直接使用0作为它们的label，会很大程度上误导CVR模型的学习。

![](../../images/models/essm_func.svg)

其中 z,y 分别表示conversion和click。注意到，在全部样本空间中，CTR对应的label为click，而CTCVR对应的label为click & conversion，这两个任务是可以使用全部样本的。因此，通过这学习两个任务，再根据上式隐式地学习CVR任务。

![esmm.png](../../images/models/esmm.png)

### 配置说明

```protobuf
model_config: {
  model_class: 'ESMM'
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
  esmm {
    groups {
      input: "user"
      dnn {
        hidden_units: [256, 128, 96, 64]
      }
    }
    groups {
      input: "item"
      dnn {
        hidden_units: [256, 128, 96, 64]
      }
    }
    groups {
      input: "combo"
      dnn {
        hidden_units: [128, 96, 64, 32]
      }
    }
    cvr_tower {
      tower_name: "cvr"
      label_name: "buy"
      dnn {
        hidden_units: [128, 96, 64, 32, 16]
      }
      num_class: 1
      weight: 1.0
      loss_type: CLASSIFICATION
      metrics_set: {
       auc {}
      }
    }
    ctr_tower {
      tower_name: "ctr"
      label_name: "clk"
      dnn {
        hidden_units: [128, 96, 64, 32, 16]
      }
      num_class: 1
      weight: 1.0
      loss_type: CLASSIFICATION
      metrics_set: {
       auc {}
      }
    }
    l2_regularization: 1e-6
  }
  embedding_regularization: 5e-5
}
```

- model\_class: 'ESMM', 不需要修改
- feature\_groups: 支持多组feature\_group
- esmm: esmm相关的参数
  - groups
    - input  tower的input必须和feature\_groups的group\_name对应
    - dnn deep part的参数配置
      - hidden\_units: dnn每一层的channel数目，即神经元的数目
  - cvr\_tower
    - tower\_name：'cvr'，不需要修改
    - label\_name: tower对应的label名，若不设置，label\_fields需与task\_towers一一对齐
    - dnn deep part的参数配置
      - hidden\_units: dnn每一层的channel数目，即神经元的数目
    - 默认为二分类任务，即num\_class默认为1，weight默认为1.0，loss\_type默认为CLASSIFICATION，metrics\_set为auc
  - ctr\_tower
    - tower\_name：'ctr'，不需要修改
    - label\_name: tower对应的label名，若不设置，label\_fields需与task\_towers一一对齐
    - dnn deep part的参数配置
      - hidden\_units: dnn每一层的channel数目，即神经元的数目
    - 默认为二分类任务，即num\_class默认为1，weight默认为1.0，loss\_type默认为CLASSIFICATION，metrics\_set为auc
- embedding\_regularization: 对embedding部分加regularization，防止overfit

### 示例Config

[ESMM\_demo.config](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/config/esmm.config)

### 参考论文

[论文地址](https://arxiv.org/abs/1804.07931)
