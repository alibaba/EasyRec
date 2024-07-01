# TextCNN

### 简介

TextCNN网络是2014年提出的用来做文本分类的卷积神经网络，由于其结构简单、效果好，在文本分类、推荐等NLP领域应用广泛。
从直观上理解，TextCNN通过一维卷积来获取句子中`N gram`的特征表示。

在推荐模型中，可以用TextCNN网络来提取文本类型的特征。

### 配置说明

```protobuf
model_config: {
  model_name: 'TextCNN'
  model_class: 'RankModel'
  feature_groups: {
    group_name: 'text_seq'
    feature_names: 'title'
    wide_deep: DEEP
  }
  backbone {
    blocks {
      name: 'text_seq'
      inputs {
        feature_group_name: 'text_seq'
      }
      input_layer {
        output_seq_and_normal_feature: true
      }
    }
    blocks {
      name: 'textcnn'
      inputs {
        block_name: 'text_seq'
      }
      keras_layer {
        class_name: 'TextCNN'
        text_cnn {
          filter_sizes: [2, 3, 4]
          num_filters: [16, 8, 8]
          pad_sequence_length: 14
          mlp {
            hidden_units: [256, 128, 64]
          }
        }
      }
    }
  }
  model_params {
    l2_regularization: 1e-6
  }
  embedding_regularization: 1e-6
}
```

- model_name: 任意自定义字符串，仅有注释作用
- model_class: 'RankModel', 不需要修改, 通过组件化方式搭建的单目标排序模型都叫这个名字
- feature_groups: 配置一组特征。
- backbone: 通过组件化的方式搭建的主干网络，[参考文档](../component/backbone.md)
  - blocks: 由多个`组件块`组成的一个有向无环图（DAG），框架负责按照DAG的拓扑排序执行个`组件块`关联的代码逻辑，构建TF Graph的一个子图
  - name/inputs: 每个`block`有一个唯一的名字（name），并且有一个或多个输入(inputs)和输出
  - keras_layer: 加载由`class_name`指定的自定义或系统内置的keras layer，执行一段代码逻辑；[参考文档](../component/backbone.md#keraslayer)
  - TextCNN: 调用TextCNN组件。组件的参数，详见[参考文档](../component/component.md#id2)
    - num_filters: 卷积核个数列表
    - filter_sizes: 卷积核步长列表
    - pad_sequence_length: 序列补齐或截断的长度
    - activation: 卷积操作的激活函数，默认为relu
  - concat_blocks: DAG的输出节点由`concat_blocks`配置项定义，如果不配置`concat_blocks`，框架会自动拼接DAG的所有叶子节点并输出。
- model_params:
  - l2_regularization: (可选) 对DNN参数的regularization, 减少overfit
- embedding_regularization: 对embedding部分加regularization, 减少overfit

### 示例Config

[text_cnn_on_movielens.config](https://github.com/alibaba/EasyRec/tree/master/samples/model_config/text_cnn_on_movielens.config)

### 参考论文

[Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
