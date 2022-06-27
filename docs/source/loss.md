# 损失函数

EasyRec支持两种损失函数配置方式：1）使用单个损失函数；2）使用多个损失函数。

## 使用单个损失函数

|损失函数|说明|
|---|---|
|CLASSIFICATION|分类Loss，二分类为sigmoid_cross_entropy；多分类为softmax_cross_entropy|
|L2_LOSS|平方损失|
|SIGMOID_L2_LOSS|对sigmoid函数的结果计算平方损失|
|CROSS_ENTROPY_LOSS|log loss 负对数损失|
|CIRCLE_LOSS|CoMetricLearningI2I模型专用|
|MULTI_SIMILARITY_LOSS|CoMetricLearningI2I模型专用|
|SOFTMAX_CROSS_ENTROPY_WITH_NEGATIVE_MINING|自动负采样版本的多分类为softmax_cross_entropy，用在二分类任务中|
|PAIR_WISE_LOSS|以优化全局AUC为目标的rank loss|
|F1_REWEIGHTED_LOSS|可以调整二分类召回率和准确率相对权重的损失函数，可有效对抗正负样本不平衡问题|


### 配置

通过`loss_type`配置项指定使用哪个具体的损失函数，默认值为`CLASSIFICATION`。

```protobuf
  {
    loss_type: L2_LOSS
  }
```

* F1_REWEIGHTED_LOSS 的参数配置

  可以调节二分类模型recall/precision相对权重的损失函数，配置如下：
  ```
  {
    loss_type: F1_REWEIGHTED_LOSS
    f1_reweight_loss {
      f1_beta_square: 0.5625
    }
  }
  ```

  - f1_beta_square: 大于1的值会导致模型更关注recall，小于1的值会导致模型更关注precision
  - F1 分数，又称平衡F分数（balanced F Score），它被定义为精确率和召回率的调和平均数。

      ![f1 score](../images/other/f1_score.svg)
  - 更一般的，我们定义 F_beta 分数为: 

      ![f_beta score](../images/other/f_beta_score.svg)
  - f1_beta_square 即为 上述公式中的 beta 系数的平方。


* SOFTMAX_CROSS_ENTROPY_WITH_NEGATIVE_MINING 
  - 支持参数配置，升级为 [support vector guided softmax loss](https://128.84.21.199/abs/1812.11317) ，
  - 目前只在DropoutNet模型中可用，可参考《 [冷启动推荐模型DropoutNet深度解析与改进](https://zhuanlan.zhihu.com/p/475117993) 》。

## 使用多个损失函数

目前所有排序模型和部分召回模型（如DropoutNet）支持同时使用多个损失函数，并且可以为每个损失函数配置不同的权重。

### 配置

下面的配置可以同时使用`F1_REWEIGHTED_LOSS`和`PAIR_WISE_LOSS`，总的loss为这两个损失函数的加权求和。

```
  losses {
    loss_type: F1_REWEIGHTED_LOSS
    weight: 1.0
  }
  losses {
    loss_type: PAIR_WISE_LOSS
    weight: 1.0
  }
  f1_reweight_loss {
    f1_beta_square: 0.5625
  }
```

排序模型同时使用多个损失函数的完整示例：
[cmbf_with_multi_loss.config](https://github.com/alibaba/EasyRec/blob/master/samples/model_config/cmbf_with_multi_loss.config)
