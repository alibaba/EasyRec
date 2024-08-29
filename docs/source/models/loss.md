# 损失函数

EasyRec支持两种损失函数配置方式：1）使用单个损失函数；2）使用多个损失函数。

### 使用单个损失函数

| 损失函数                                       | 说明                                                         |
| ------------------------------------------ | ---------------------------------------------------------- |
| CLASSIFICATION                             | 分类Loss，二分类为sigmoid_cross_entropy；多分类为softmax_cross_entropy |
| L2_LOSS                                    | 平方损失                                                       |
| SIGMOID_L2_LOSS                            | 对sigmoid函数的结果计算平方损失                                        |
| CROSS_ENTROPY_LOSS                         | log loss 负对数损失                                             |
| CIRCLE_LOSS                                | CoMetricLearningI2I模型专用                                    |
| MULTI_SIMILARITY_LOSS                      | CoMetricLearningI2I模型专用                                    |
| SOFTMAX_CROSS_ENTROPY_WITH_NEGATIVE_MINING | 自动负采样版本的多分类softmax_cross_entropy，用在二分类任务中                  |
| BINARY_FOCAL_LOSS                          | 支持困难样本挖掘和类别平衡的focal loss                                   |
| PAIR_WISE_LOSS                             | 以优化全局AUC为目标的rank loss                                      |
| PAIRWISE_FOCAL_LOSS                        | pair粒度的focal loss, 支持自定义pair分组                             |
| PAIRWISE_LOGISTIC_LOSS                     | pair粒度的logistic loss, 支持自定义pair分组                          |
| JRC_LOSS                                   | 二分类 + listwise ranking loss                                |
| F1_REWEIGHTED_LOSS                         | 可以调整二分类召回率和准确率相对权重的损失函数，可有效对抗正负样本不平衡问题                     |
| ORDER_CALIBRATE_LOSS                       | 使用目标依赖关系校正预测结果的辅助损失函数，详见[AITM](aitm.md)模型                  |

- 说明：SOFTMAX_CROSS_ENTROPY_WITH_NEGATIVE_MINING
  - 支持参数配置，升级为 [support vector guided softmax loss](https://128.84.21.199/abs/1812.11317) ，
  - 目前只在DropoutNet模型中可用，可参考《 [冷启动推荐模型DropoutNet深度解析与改进](https://zhuanlan.zhihu.com/p/475117993) 》。

#### 配置

通过`loss_type`配置项指定使用哪个具体的损失函数，默认值为`CLASSIFICATION`。

```protobuf
  {
    loss_type: L2_LOSS
  }
```

### 使用多个损失函数

目前所有排序模型，包括多目标模型(`ESMM`模型除外)，和部分召回模型（如DropoutNet）支持同时使用多个损失函数，并且可以为每个损失函数配置不同的权重。

##### 配置

下面的配置可以同时使用`F1_REWEIGHTED_LOSS`和`PAIR_WISE_LOSS`，总的loss为这两个损失函数的加权求和。

```
  losses {
    loss_type: F1_REWEIGHTED_LOSS
    weight: 1.0
    f1_reweighted_loss {
      f1_beta_square: 0.5625
    }
  }
  losses {
    loss_type: PAIR_WISE_LOSS
    weight: 1.0
  }
```

- F1_REWEIGHTED_LOSS 的参数配置

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
    - ![f1 score](../../images/other/f1_score.svg)
  - 更一般的，我们定义 F_beta 分数为:
    - ![f_beta score](../../images/other/f_beta_score.svg)
  - f1_beta_square 即为 上述公式中的 beta 系数的平方。

- PAIRWISE_FOCAL_LOSS 的参数配置

  - gamma: focal loss的指数，默认值2.0
  - alpha: 调节样本权重的类别平衡参数，建议根据正负样本比例来配置alpha，即 alpha / (1-alpha) = #Neg / #Pos
  - session_name: pair分组的字段名，比如user_id
  - hinge_margin: 当pair的logit之差大于该参数值时，当前样本的loss为0，默认值为1.0
  - ohem_ratio: 困难样本的百分比，只有部分困难样本参与loss计算，默认值为1.0
  - temperature: 温度系数，logit除以该参数值后再参与计算，默认值为1.0

- PAIRWISE_LOGISTIC_LOSS 的参数配置

  - session_name: pair分组的字段名，比如user_id
  - hinge_margin: 当pair的logit之差大于该参数值时，当前样本的loss为0，默认值为1.0
  - ohem_ratio: 困难样本的百分比，只有部分困难样本参与loss计算，默认值为1.0
  - temperature: 温度系数，logit除以该参数值后再参与计算，默认值为1.0

- PAIRWISE_LOSS 的参数配置

  - session_name: pair分组的字段名，比如user_id
  - margin: 当pair的logit之差减去该参数值后再参与计算，即正负样本的logit之差至少要大于margin，默认值为0
  - temperature: 温度系数，logit除以该参数值后再参与计算，默认值为1.0

备注：上述 PAIRWISE\_\*\_LOSS 都是在mini-batch内构建正负样本pair，目标是让正负样本pair的logit相差尽可能大

- BINARY_FOCAL_LOSS 的参数配置

  - gamma: focal loss的指数，默认值2.0
  - alpha: 调节样本权重的类别平衡参数，建议根据正负样本比例来配置alpha，即 alpha / (1-alpha) = #Neg / #Pos
  - ohem_ratio: 困难样本的百分比，只有部分困难样本参与loss计算，默认值为1.0
  - label_smoothing: 标签平滑系数

- JRC_LOSS 的参数配置

  - alpha: ranking loss 与 calibration loss 的相对权重系数；不设置该值时，触发权重自适应学习
  - session_name: list分组的字段名，比如user_id
  - 参考论文：《 [Joint Optimization of Ranking and Calibration with Contextualized Hybrid Model](https://arxiv.org/pdf/2208.06164.pdf) 》
  - 使用示例: [dbmtl_with_jrc_loss.config](https://github.com/alibaba/EasyRec/blob/master/samples/model_config/dbmtl_on_taobao_with_multi_loss.config)

排序模型同时使用多个损失函数的完整示例：
[cmbf_with_multi_loss.config](https://github.com/alibaba/EasyRec/blob/master/samples/model_config/cmbf_with_multi_loss.config)

多目标排序模型同时使用多个损失函数的完整示例:
[dbmtl_with_multi_loss.config](https://github.com/alibaba/EasyRec/blob/master/samples/model_config/dbmtl_on_taobao_with_multi_loss.config)

### Loss权重自适应

多目标学习任务中，人工指定多个损失函数的固定权重通常不能获得最好的效果。EasyRec支持损失函数权重自适应学习，示例如下：

```protobuf
  loss_weight_strategy: Uncertainty
  losses {
    loss_type: CLASSIFICATION
    learn_loss_weight: true
  }
  losses {
    loss_type: BINARY_FOCAL_LOSS
    learn_loss_weight: true
    binary_focal_loss {
      gamma: 2.0
      alpha: 0.85
    }
  }
  losses {
    loss_type: PAIRWISE_FOCAL_LOSS
    learn_loss_weight: true
    pairwise_focal_loss {
      session_name: "client_str"
      hinge_margin: 1.0
    }
  }
```

通过`learn_loss_weight`参数配置是否需要开启权重自适应学习，默认不开启。开启之后，`weight`参数不再生效。

- loss_weight_strategy: Uncertainty
  - 表示通过不确定性来度量损失函数的权重；目前在`learn_loss_weight: true`时必须要设置该值
- loss_weight_strategy: Random
  - 表示损失函数的权重设定为归一化的随机数

### 参考论文：

- 《 Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics 》
- 《 [Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning](https://arxiv.org/abs/2111.10603) 》
- [AITM: Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising](https://arxiv.org/pdf/2105.08489.pdf)
