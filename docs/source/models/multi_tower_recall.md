# MultiTowerRecall

### 简介

专为负采样和序列特征训练准备的双塔召回模型，分为user塔和item塔。
注：使用时需指定user id和item id。

### 配置说明

```protobuf
model_config:{
  model_class: "MultiTowerRecall"
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
    wide_deep:DEEP
    negative_sampler:true
    sequence_features: {
      group_name: "seq_fea"
      allow_key_search: true
      need_key_feature:true
      seq_att_map: {
        key: "brand"
        key: "cate_id"
        hist_seq: "tag_brand_list"
        hist_seq: "tag_category_list"
      }
    }
  }
  feature_groups: {
    group_name: "item"
    feature_names: 'adgroup_id'
    feature_names: 'cate_id'
    feature_names: 'campaign_id'
    feature_names: 'customer'
    feature_names: 'brand'
    wide_deep:DEEP
  }
  multi_tower_recall {
    user_tower {
      id: "user_id"
      dnn {
        hidden_units: [256, 128, 64, 32]
        # dropout_ratio : [0.1, 0.1, 0.1, 0.1]
      }
    }
    item_tower {
      id: "adgroup_id"
      dnn {
        hidden_units: [256, 128, 64, 32]
      }
    }
    final_dnn {
      hidden_units: [128, 96, 64, 32, 16]
    }
    l2_regularization: 1e-6
  }
  loss_type: CLASSIFICATION
  embedding_regularization: 5e-6
}
```

- model_class: 'MultiTowerRecall', 不需要修改
- feature_groups: 需要两个feature_group: user和item, **group name不能变**
- multi_tower_recall: multi_tower_recall相关的参数，必须配置user_tower和item_tower
- user_tower/item_tower:
  - dnn: deep part的参数配置
    - hidden_units: dnn每一层的channel数目，即神经元的数目
- embedding_regularization: 对embedding部分加regularization，防止overfit

支持的metric_set包括:

- auc
- mean_absolute_error
- accuracy

### 示例Config

见路径：samples/model_config/multi_tower_recall_neg_sampler_sequence_feature.config
