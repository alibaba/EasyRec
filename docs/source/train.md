# 训练

## train_config

- log_step_count_steps: 200    # 每200轮打印一行log

- optimizer_config     # 优化器相关的参数

  ```protobuf
  {
    adam_optimizer: {
      learning_rate: {
         exponential_decay_learning_rate {
            initial_learning_rate: 0.0001
            decay_steps: 100000
            decay_factor: 0.5
            min_learning_rate: 0.0000001
         }
    }
  }
  ```

- sync_replicas: true  # 是否同步训练，默认是false

  - 使用SyncReplicasOptimizer进行分布式训练(同步模式)
  - 仅在train_distribute为NoStrategy时可以设置成true，其它情况应该设置为false
  - PS异步训练也设置为false

- train_distribute: 默认不开启Strategy(NoStrategy), strategy确定分布式执行的方式

  - NoStrategy 不使用Strategy
  - PSStrategy 异步ParameterServer模式
  - MirroredStrategy 单机多卡模式，仅在PAI上可以使用，本地和EMR上不能使用
  - MultiWorkerMirroredStrategy 多机多卡模式，在TF版本>=1.15时可以使用

- num_gpus_per_worker: 仅在MirrorredStrategy, MultiWorkerMirroredStrategy, PSStrategy的时候有用

- num_steps: 1000

  - 总共训练多少轮
  - num_steps = total_sample_num * num_epochs / batch_size / num_workers
  - **分布式训练时一定要设置num_steps，否则评估任务会结束不了**

- fine_tune_checkpoint: 需要restore的checkpoint路径，也可以是包含checkpoint的目录，如果目录里面有多个checkpoint，将使用最新的checkpoint

- fine_tune_ckpt_var_map: 需要restore的参数列表文件路径，文件的每一行是{variable_name in current model ckpt}\\t{variable name in old model ckpt}

  - 需要设置fine_tune_ckpt_var_map的情形:
    - current ckpt和old ckpt不完全匹配, 如embedding的名字不一样:
      - old: input_layer/shopping_level_embedding/embedding_weights
      - new: input_layer/shopping_embedding/embedding_weights
    - 仅需要restore old ckpt里面的部分variable, 如embedding_weights
  - 可以通过下面的文件查看参数列表

  ```python
  import tensorflow as tf
  import os, sys

  ckpt_reader = tf.train.NewCheckpointReader('experiments/model.ckpt-0')
  ckpt_var2shape_map = ckpt_reader.get_variable_to_shape_map()
  for key in ckpt_var2shape_map:
    print(key)
  ```

- save_checkpoints_steps: 每隔多少轮保存一次checkpoint, 默认是1000

- save_checkpoints_secs: 每隔多少s保存一次checkpoint, 不可以和save_checkpoints_steps同时指定

- keep_checkpoint_max: 最多保存多少个checkpoint, 默认是10

- log_step_count_steps: 每隔多少轮，打印一次训练信息，默认是10

- save_summary_steps: 每隔多少轮，保存一次summary信息，默认是1000

- 更多参数请参考[easy_rec/python/protos/train.proto](./reference.md)

### 损失函数

EasyRec支持两种损失函数配置方式：1）使用单个损失函数；2）使用多个损失函数。

#### 使用单个损失函数

| 损失函数                                       | 说明                                                         |
| ------------------------------------------ | ---------------------------------------------------------- |
| CLASSIFICATION                             | 分类Loss，二分类为sigmoid_cross_entropy；多分类为softmax_cross_entropy |
| L2_LOSS                                    | 平方损失                                                       |
| SIGMOID_L2_LOSS                            | 对sigmoid函数的结果计算平方损失                                        |
| CROSS_ENTROPY_LOSS                         | log loss 负对数损失                                             |
| CIRCLE_LOSS                                | CoMetricLearningI2I模型专用                                    |
| MULTI_SIMILARITY_LOSS                      | CoMetricLearningI2I模型专用                                    |
| SOFTMAX_CROSS_ENTROPY_WITH_NEGATIVE_MINING | 自动负采样版本的多分类softmax_cross_entropy，用在二分类任务中                 |
| PAIR_WISE_LOSS                             | 以优化全局AUC为目标的rank loss                                      |
| F1_REWEIGHTED_LOSS                         | 可以调整二分类召回率和准确率相对权重的损失函数，可有效对抗正负样本不平衡问题                     |

* 说明：SOFTMAX_CROSS_ENTROPY_WITH_NEGATIVE_MINING
  - 支持参数配置，升级为 [support vector guided softmax loss](https://128.84.21.199/abs/1812.11317) ，
  - 目前只在DropoutNet模型中可用，可参考《 [冷启动推荐模型DropoutNet深度解析与改进](https://zhuanlan.zhihu.com/p/475117993) 》。

##### 配置

通过`loss_type`配置项指定使用哪个具体的损失函数，默认值为`CLASSIFICATION`。

```protobuf
  {
    loss_type: L2_LOSS
  }
```

#### 使用多个损失函数

目前所有排序模型，包括多目标模型，和部分召回模型（如DropoutNet）支持同时使用多个损失函数，并且可以为每个损失函数配置不同的权重。

##### 配置

下面的配置可以同时使用`F1_REWEIGHTED_LOSS`和`PAIR_WISE_LOSS`，总的loss为这两个损失函数的加权求和。

```
  losses {
    loss_type: F1_REWEIGHTED_LOSS
    weight: 1.0
    f1_reweight_loss {
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
    - ![f1 score](../images/other/f1_score.svg)
  - 更一般的，我们定义 F_beta 分数为:
    - ![f_beta score](../images/other/f_beta_score.svg)
  - f1_beta_square 即为 上述公式中的 beta 系数的平方。

排序模型同时使用多个损失函数的完整示例：
[cmbf_with_multi_loss.config](https://github.com/alibaba/EasyRec/blob/master/samples/model_config/cmbf_with_multi_loss.config)

## 训练命令

### Local

```bash
python -m easy_rec.python.train_eval --pipeline_config_path dwd_avazu_ctr_deepmodel.config
```

- --pipeline_config_path: config文件路径
- --continue_train: restore之前的checkpoint，继续训练
- --model_dir: 如果指定了model_dir将会覆盖config里面的model_dir，一般在周期性调度的时候使用
- --edit_config_json: 使用json的方式对config的一些字段进行修改，如:
  ```sql
  --edit_config_json='{"train_config.fine_tune_checkpoint": "oss://easyrec/model.ckpt-50"}'
  ```

### On PAI

```sql
pai -name easy_rec_ext -project algo_public
-Dconfig=oss://easyrec/easy_rec_test/dwd_avazu_ctr_deepmodel_ext.config
-Dcmd=train
-Dtrain_tables=odps://pai_online_project/tables/dwd_avazu_ctr_deepmodel_train
-Deval_tables=odps://pai_online_project/tables/dwd_avazu_ctr_deepmodel_test
-Dcluster='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":3, "cpu":1000, "gpu":100, "memory":40000}}'
-Darn=acs:ram::xxx:role/ev-ext-test-oss
-Dbuckets=oss://easyrec/
-DossHost=oss-cn-beijing-internal.aliyuncs.com
-Deval_method=separate;
```

- -Dtrain_tables: 训练表，可以指定多个，逗号分隔
- -Deval_tables: 评估表，可以指定多个，逗号分隔
- -Dcluster: 定义PS的数目和worker的数目，如果设置了--eval_method=separate，有一个worker将被用于做评估
- -Dconfig: 训练用的配置文件
- -Dcmd: train   模型训练
- -Deval_method: 训练时需要评估, 可选参数:
  - separate: 有一个worker被单独用来做评估(不参与训练)
  - none: 不需要评估
  - master: 在master结点上做评估，master结点也参与训练
- -Darn: rolearn  注意这个的arn要替换成客户自己的。可以从dataworks的设置中查看arn。
- -DossHost: ossHost地址
- -Dbuckets: config所在的bucket和保存模型的bucket; 如果有多个bucket，逗号分割
- -Dselected_cols 表里面用于训练和评估的列, 有助于提高训练速度
- -Dmodel_dir: 如果指定了model_dir将会覆盖config里面的model_dir，一般在周期性调度的时候使用。
- -Dedit_config_json: 使用json的方式对config的一些字段进行修改，如:
  ```sql
  -Dedit_config_json='{"train_config.fine_tune_checkpoint": "oss://easyrec/model.ckpt-50"}'
  ```
- 如果是pai内部版,则不需要指定arn和ossHost, arn和ossHost放在-Dbuckets里面
  - -Dbuckets=oss://easyrec/?role_arn=acs:ram::xxx:role/ev-ext-test-oss&host=oss-cn-beijing-internal.aliyuncs.com

### On EMR

单机单卡模式:

```bash
el_submit -t standalone -a easy_rec_train -f dwd_avazu_ctr_deepmodel.config -m local  -wn 1 -wc 6 -wm 20000  -wg 1 -c "python -m easy_rec.python.train_eval --pipeline_config_path dwd_avazu_ctr_deepmodel.config --continue_train"
```

- 参数同Local模式

多worker模式:

- 需要在配置文件中设置train_config.train_distribute为MultiWorkerMirroredStrategy

```bash
el_submit -t standalone -a easy_rec_train -f dwd_avazu_ctr_deepmodel.config -m local  -wn 1 -wc 6 -wm 20000  -wg 2 -c "python -m easy_rec.python.train_eval --pipeline_config_path dwd_avazu_ctr_deepmodel.config --continue_train"
```

- 参数同Local模式

PS模式:

- 需要在配置文件中设置train_config.sync_replicas为true

```bash
el_submit -t tensorflow-ps -a easy_rec_train -f dwd_avazu_ctr_deepmodel.config -m local -pn 1 -pc 4 -pm 20000 -wn 3 -wc 6 -wm 20000 -c "python -m easy_rec.python.train_eval --pipeline_config_path dwd_avazu_ctr_deepmodel.config --continue_train"
```

- 参数同Local模式
