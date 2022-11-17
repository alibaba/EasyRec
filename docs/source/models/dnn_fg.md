# MultiTowerRecall

### 简介

专为接入RTP FG时加入负采样和序列特征训练准备的DNN模型。

### 配置说明

```protobuf
fg_json_path: "!samples/model_config/fg_fusion_train_seq.json"
model_config {
  model_class: "DNNFG"
  feature_groups {
    group_name: "all"
    feature_names: 'adgroup_id'
    feature_names: 'cate_id'
    feature_names: 'campaign_id'
    feature_names: 'customer'
    feature_names: 'brand'
    feature_names: 'user_id'
    feature_names: 'cms_segid'
    feature_names: 'cms_group_id'
    feature_names: 'final_gender_code'
    feature_names: 'age_level'
    feature_names: 'pvalue_level'
    feature_names: 'shopping_level'
    feature_names: 'occupation'
    feature_names: 'new_user_class_level'
    wide_deep: DEEP
    sequence_features: {
      group_name: "seq_fea"
      tf_summary: false
      seq_att_map: {
        key: "cate_id"
        key: "brand"
        hist_seq: "click_seq__cate_id"
        hist_seq: "click_seq__brand"
      }
    }
  }
  dnnfg {
    dnn {
      hidden_units: 256
      hidden_units: 128
      hidden_units: 64
    }
    l2_regularization: 1e-6
  }
  embedding_regularization: 5e-6
}
```

- fg_json_path: 指定fg json文件目录
- model_class: 'DNNFG', 不需要修改
- feature_groups: 需要一个feature_group: all, **group name不能变**
- dnnfg: dnnfg相关的参数
  - dnn: deep part的参数配置
    - hidden_units: dnn每一层的channel数目，即神经元的数目
- embedding_regularization: 对embedding部分加regularization，防止overfit

支持的metric_set包括:

- auc
- gauc
- recall_at_topK

### 示例Config

见路径：samples/model_config/fg_fusion_train_neg_seq_on_dnn.config
