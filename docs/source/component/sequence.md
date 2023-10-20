# 序列化组件的配置方式

序列模型（DIN、BST）的组件化配置方式需要把输入特征放置在同一个`feature_group`内。

序列模型一般包含 `history behavior sequence` 与 `target item` 两部分，且每部分都可能包含多个属性(子特征)。

在序列组件输入的`feature_group`内，**按照顺序**定义 `history behavior sequence` 与 `target item`的各个子特征。

框架按照特征定义的类型`feature_type`字段来识别某个具体的特征是属于 `history behavior sequence` 还是 `target item`。
所有 `SequenceFeature` 类型的子特征都被识别为`history behavior sequence`的一部分; 所有非`SequenceFeature` 类型的子特征都被识别为`target item`的一部分。

**两部分的子特征的顺序需要保持一致**。在下面的例子中，
- `concat([cate_id,brand], axis=-1)` 是`target item`最终的embedding（2D）;
- `concat([tag_category_list, tag_brand_list], axis=-1)` 是`history behavior sequence`最终的embedding（3D）

```protobuf
model_config: {
  model_name: 'DIN'
  model_class: 'RankModel
  ...
  feature_groups: {
    group_name: 'sequence'
    feature_names: "cate_id"
    feature_names: "brand"
    feature_names: "tag_category_list"
    feature_names: "tag_brand_list"
    wide_deep: DEEP
  }
  backbone {
    blocks {
      name: 'seq_input'
      inputs {
        feature_group_name: 'sequence'
      }
      input_layer {
        output_seq_and_normal_feature: true
      }
    }
    blocks {
      name: 'DIN'
      inputs {
        block_name: 'seq_input'
      }
      keras_layer {
        class_name: 'DIN'
        din {
          attention_dnn {
            hidden_units: 32
            hidden_units: 1
            activation: "dice"
          }
          need_target_feature: true
        }
      }
    }
    ...
  }
}
```

使用序列组件时，必须配置一个`input_layer`类型的`block`，并且配置`output_seq_and_normal_feature: true`参数，如下。

```protobuf
blocks {
  name: 'seq_input'
  inputs {
    feature_group_name: 'sequence'
  }
  input_layer {
    output_seq_and_normal_feature: true
  }
}
```

## 完整的例子

- [DIN](../models/din.md)
- [BST](../models/bst.md)
