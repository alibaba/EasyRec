# 自定义辅助损失函数组件

可以使用如下方法添加多个辅助损失函数。

在`easy_rec/python/layers/keras/auxiliary_loss.py`里添加一个新的loss函数。
如果计算逻辑比较复杂，建议在一个单独的python文件中实现，然后在`auxiliary_loss.py`里import并使用。

注意：用来标记损失函数类型的`loss_type`参数需要全局唯一。

## 配置方法

```protobuf
blocks {
  name: 'custom_loss'
  inputs {
    block_name: 'pred'
  }
  inputs {
    block_name: 'logit'
  }
  merge_inputs_into_list: true
  keras_layer {
    class_name: 'AuxiliaryLoss'
    st_params {
      fields {
        key: "loss_type"
        value { string_value: "my_custom_loss" }
      }
    }
  }
}
```

st_params 参数列表下可以追加自定义参数。

记得使用`concat_blocks`或者`output_blocks`配置输出的block列表（不包括当前`custom_loss`节点）。
