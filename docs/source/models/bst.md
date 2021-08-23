# BST

### 简介

利用近年因 Transformer 而备受关注的 Multi-head Self-attention，捕捉用户行为序列的序列信息。支持多组序列共同embedding，如hist_item_id, hist_category_id。目前结合multitower共同使用，bst部分作为multitower的一个塔。

### 模型配置

```protobuf
model_config:{
  model_class: "MultiTowerBST"
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
    feature_names: 'pid'
    wide_deep: DEEP
  }
  seq_att_groups: {
    group_name: "bst"
    seq_att_map: {
       key: "brand"
       hist_seq: "tag_brand_list"
    }
    seq_att_map: {
       key: "cate_id"
       hist_seq: "tag_category_list"
    }
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
    bst_towers {
      input: "bst"
      seq_len: 50
      multi_head_size: 4
    }
    final_dnn {
      hidden_units: [128, 96, 64, 32, 16]
    }
    l2_regularization: 5e-7
  }
  embedding_regularization: 5e-5
}

```

- model_class: 'MultiTowerBST', 不需要修改。
- feature_groups: 可配置多个feature_group，group name可以变。
- seq_att_groups: 可配置多个seq_att_groups。
  - group name
  - seq_att_map: 需配置key和hist_seq，一一对应。
- multi_tower: multi_tower相关的参数。
  - towers: 每个feature_group对应了一个tower。
    - input必须和feature_groups的group_name对应
    - dnn: deep part的参数配置
      - hidden_units: dnn每一层的channel数目，即神经元的数目
  - bst_towers: 每个seq_att_groups对应了一个bst_tower。
    - input必须和seq_att_groups的group_name对应
    - seq_len: 历史序列的最大长度
    - multi_head_size: Multi-head Self-attention 中的 head size
  - final_dnn 整合towers和din_towers的输入。
    - hidden_units: dnn每一层的channel数目，即神经元的数目
- embedding_regularization: 对embedding部分加regularization，防止overfit

### 示例config

[BST_demo.config](https://easyrec.oss-cn-beijing.aliyuncs.com/config/bst.config)

### 参考论文

[Behavior Sequence Transformer](https://arxiv.org/abs/1905.06874v1)
