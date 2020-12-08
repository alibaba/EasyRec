# SimpleMultiTask

### 简介

针对简单的多任务模型,所有任务共享特征和embedding,但是针对每个任务使用单独的Task Tower,任务之间相互独立

### 配置说明

```protobuf
model_config:{
  model_class: "SimpleMultiTask"
  feature_groups {
    group_name: "all"
    feature_names: "user_id"
    feature_names: "cms_segid"
    ...
    feature_names: "tag_brand_list"
    wide_deep: DEEP
  }

  simple_multi_task {
    task_towers {
      tower_name: "ctr"
      label_name: "clk"
      dnn {
        hidden_units: [256, 192, 128, 64]
      }
      num_class: 1
      weight: 1.0
      loss_type: CLASSIFICATION
      metrics_set: {
       auc {}
      }
    }
    task_towers {
      tower_name: "cvr"
      label_name: "buy"
      dnn {
        hidden_units: [256, 192, 128, 64]
      }
      num_class: 1
      weight: 1.0
      loss_type: CLASSIFICATION
      metrics_set: {
       auc {}
      }
    }
    l2_regularization: 0.0
  }
  embedding_regularization: 0.0
}
```

- model\_class: 'SimpleMultiTask', 不需要修改
- feature\_groups: 配置一个名为'all'的feature\_group。
- simple\_multi\_task: 相关的参数
  - task\_towers 根据任务数配置task\_towers
    - tower\_name：任务名
    - label\_name: tower对应的label名，若不设置，label\_fields需与task\_towers一一对齐
    - dnn deep part的参数配置
      - hidden\_units: dnn每一层的channel数目，即神经元的数目
    - 默认为二分类任务，即num\_class默认为1，weight默认为1.0，loss\_type默认为CLASSIFICATION，metrics\_set为auc
  - embedding\_regularization: 对embedding部分加regularization，防止overfit
