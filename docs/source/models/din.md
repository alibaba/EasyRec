# DIN

### 简介

利用DIN算法建模用户点击序列。支持多组序列共同embedding，如hist\_item\_id, hist\_category\_id。目前结合multitower共同使用，din部分作为multitower的一个塔。
![din.png](../../images/models/din.png)

### 模型配置

```protobuf
model_config: {
  model_class: 'MultiTowerDIN'
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
    feature_names: 'pid'
    wide_deep: DEEP
  }
  seq_att_groups: {
    group_name: "din"
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
    din_towers {
      input: "din"
      dnn {
        hidden_units: [128, 64, 32, 1]
      }
    }
    final_dnn {
      hidden_units: [128, 96, 64, 32, 16]
    }
    l2_regularization: 5e-7
  }
  embedding_regularization: 5e-5
}

```

- model\_class: 'MultiTowerDIN', 不需要修改。
- feature\_groups: 可配置多个feature\_group，group name可以变。
- seq\_att\_groups: 可配置多个seq\_att\_groups。
  - group name
  - seq\_att\_map: 需配置key和hist\_seq，一一对应。
- multi\_tower: multi\_tower相关的参数
  - towers: 每个feature\_group对应了一个tower。
    - input必须和feature\_groups的group\_name对应。
    - dnn: deep part的参数配置
      - hidden\_units: dnn每一层的channel数目，即神经元的数目
  - din\_towers: 每个seq\_att\_groups对应了一个din\_tower
    - input必须和seq\_att\_groups的group\_name对应。
    - dnn: deep part的参数配置
      - hidden\_units: dnn每一层的channel数目，即神经元的数目
  - final\_dnn 整合towers和din\_towers的输入
    - hidden\_units: dnn每一层的channel数目，即神经元的数目
- embedding\_regularization: 对embedding部分加regularization，防止overfit

### 示例config

[DIN\_demo.config](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/config/din.config)

### 参考论文

[Deep Interest Network](https://arxiv.org/abs/1706.06978)
