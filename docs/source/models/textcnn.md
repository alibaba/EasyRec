# TextCNN

### 简介

TextCNN网络是2014年提出的用来做文本分类的卷积神经网络，由于其结构简单、效果好，在文本分类、推荐等NLP领域应用广泛。
对于文本分类问题，常见的方法无非就是抽取文本的特征。然后再基于抽取的特征训练一个分类器。 然而研究证明，TextCnn在文本分类问题上有着更加卓越的表现。
从直观上理解，TextCNN通过一维卷积来获取句子中n-gram的特征表示。
TextCNN对文本浅层特征的抽取能力很强，在短文本领域专注于意图分类时效果很好，应用广泛，且速度较快。

Yoon Kim在论文[EMNLP 2014][Convolutional neural networks for sentence classication](https://www.aclweb.org/anthology/D14-1181.pdf)
提出了TextCNN并给出基本的结构。
将卷积神经网络CNN应用到文本分类任务，利用多个不同size的kernel来提取句子中的关键信息（类似于多窗口大小的ngram），从而能够更好地捕捉局部相关性。
模型的主体结构如图所示：

![din.png](../../images/models/text_cnn.png)

### 模型配置

```protobuf
feature_configs: {
  input_names: 'title'
  feature_type: SequenceFeature
  separator: ' '
  embedding_dim: 32
  hash_bucket_size: 10000
  sequence_combiner: {
    text_cnn: {
      filter_sizes: [2, 3, 4]
      num_filters: [16, 8, 8]
    }
  }
}
model_config: {
  model_class: 'MultiTower'
  feature_groups: {
    group_name: 'item'
    feature_names: 'title'
    wide_deep: DEEP
  }

  multi_tower {
    towers {
      input: "item"
      dnn {
        hidden_units: [64]
      }
    }
    final_dnn {
      hidden_units: []
    }
    l2_regularization: 1e-6
  }
  embedding_regularization: 1e-4
}
```

- model_class: 'MultiTower', 不需要修改。
- feature_groups: 特征组，group name可以变。
  - feature_names: 配置需要使用text cnn layer的特征名
  - feature_config: 特征类型为`SequenceFeature`, `sequence_combiner`必须配置为`text_cnn`；同时指定卷积核的个数和步长。
- multi_tower: multi_tower相关的参数，这里借用MultiTower模型框架，通过只配置一个Tower的方式来构建text cnn layer之后的MLP层。
  - towers: 每个feature_group对应了一个tower。
    - input必须和feature_groups的group_name对应。
    - dnn: deep part的参数配置
      - hidden_units: dnn每一层的unit数目，即神经元的数目
  - final_dnn 整合towers和din_towers的输入
    - hidden_units: 这里配置为空即可
- embedding_regularization: 对embedding部分加regularization，防止overfit

### 示例config

[TextCNN_demo.config](https://easyrec.oss-cn-beijing.aliyuncs.com/config/text_cnn.config)

### 参考论文

[Text CNN](https://www.aclweb.org/anthology/D14-1181.pdf)
