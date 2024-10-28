# 训练

## train_config

- log_step_count_steps: 200    # 每200步打印一行log

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

  - 多优化器支持:
    - 可以配置两个optimizer, 分别对应embedding权重和dense权重;
    - 实现参考EasyRecModel.get_grouped_vars和multi_optimizer.MultiOptimizer;
    - 示例(samples/model_config/deepfm_combo_on_avazu_embed_adagrad.config):
      ```protobuf
      train_config {
        ...
         optimizer_config {  # for embedding_weights
           adagrad_optimizer {
             learning_rate {
               constant_learning_rate {
                 learning_rate: 0.05
               }
             }
             initial_accumulator_value: 1.0
           }
         }

         optimizer_config: {  # for dense weights
           adam_optimizer: {
             learning_rate: {
               exponential_decay_learning_rate {
                 initial_learning_rate: 0.0001
                 decay_steps: 10000
                 decay_factor: 0.5
                 min_learning_rate: 0.0000001
               }
             }
           }
         }
      ```
    - Note: [WideAndDeep](./models/wide_and_deep.md)模型的optimizer设置:
      - 设置两个optimizer时, 第一个optimizer仅用于wide参数;
      - 如果要给deep embedding单独设置optimizer, 需要设置3个optimizer.

- sync_replicas: true  # 是否同步训练，默认是false

  - 使用SyncReplicasOptimizer进行分布式训练(同步模式)
  - 仅在train_distribute为NoStrategy时可以设置成true，其它情况应该设置为false
  - PS异步训练也设置为false

- train_distribute: 默认不开启Strategy(NoStrategy), strategy确定分布式执行的方式, 可以分成两种模式: PS-Worker模式 和 All-Reduce模式

  - PS-Worker模式:
    - NoStrategy: 根据sync_replicas的取值决定采用同步或者异步训练
      - sync_replicas=true，采用ps worker同步训练
        - 注意: 该模式容易导致ps存在通信瓶颈, 建议用混合并行的模式进行同步训练
      - sync_replicas=false, 采用ps worker异步训练
  - All-Reduce模式:
    - 数据并行:
      - MirroredStrategy: 单机多卡模式，仅在PAI上可以使用，本地和EMR上不能使用
      - MultiWorkerMirroredStrategy: 多机多卡模式，在TF版本>=1.15时可以使用
      - HorovodStragtegy: horovod多机多卡并行, 需要安装horovod
    - 混合并行: 数据并行 + Embedding分片, 需要安装horovod
      - EmbeddingParallelStrategy: 在horovod多机多卡并行的基础上, 增加了Embedding分片的功能
      - SokStrategy: 在horovod多机多卡并行的基础上, 增加了[SOK](https://github.com/NVIDIA-Merlin/HugeCTR/tree/main/sparse_operation_kit) Key-Value Embedding和Embedding分片的功能
        - 注意: 该模式仅支持GPU模式, 需要安装SOK.

- num_gpus_per_worker: 仅在MirrorredStrategy, MultiWorkerMirroredStrategy, PSStrategy的时候有用

- num_steps: 1000

  - 总共训练多少轮
  - num_steps = total_sample_num * num_epochs / batch_size / num_workers

- fine_tune_checkpoint: 需要restore的checkpoint路径，也可以是包含checkpoint的目录，如果目录里面有多个checkpoint，将使用最新的checkpoint

- fine_tune_ckpt_var_map: 需要restore的参数列表文件路径，文件的每一行是{variable_name in current model}\\t{variable name in old model ckpt}

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

- save_checkpoints_steps: 每隔多少步保存一次checkpoint, 默认是1000。当训练数据量很大的时候，这个值要设置大一些

- save_checkpoints_secs: 每隔多少s保存一次checkpoint, 不可以和save_checkpoints_steps同时指定

- keep_checkpoint_max: 最多保存多少个checkpoint, 默认是10。当模型较大的时候可以设置为5，可节约存储

- log_step_count_steps: 每隔多少轮，打印一次训练信息，默认是10

- save_summary_steps: 每隔多少轮，保存一次summary信息，默认是1000

- 更多参数请参考[easy_rec/python/protos/train.proto](./reference.md)

## 训练命令

### Local

```bash
python -m easy_rec.python.train_eval --pipeline_config_path dwd_avazu_ctr_deepmodel.config
```

- --pipeline_config_path: config文件路径
- --continue_train: restore之前的checkpoint，继续训练
- --model_dir: 如果指定了model_dir将会覆盖config里面的model_dir，一般在周期性调度的时候使用
- --edit_config_json: 使用json的方式对config的一些字段进行修改，如:
  ```bash
  --edit_config_json='{"train_config.fine_tune_checkpoint": "experiments/ctr/model.ckpt-50"}'
  ```
- Extend Args: 命令行参数修改config, 类似edit_config_json
  - 支持train_config.*, eval_config.*, data_config.*, feature_config.*
  - 示例:
  ```bash
  --train_config.fine_tune_checkpoint=experiments/ctr/model.ckpt-50
  --data_config.negative_sampler.input_path=data/test/tb_data/taobao_ad_feature_gl
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

### On DLC

- 基于Kubeflow的云原生的训练方式
- [参考文档](./quick_start/dlc_tutorial.md)

### On EMR

- 基于开源大数据平台的训练方式
- [参考文档](https://help.aliyun.com/zh/emr/emr-on-ecs/user-guide/use-easyrec-to-perform-model-training-evaluation-and-prediction-on-data-from-hive-tables)

## 混合并行(EmbeddingParallel)

混合并行模式下Embedding参数会分片, 均匀分布到各个worker上, 通过all2all的通信方式来聚合不同worker上的Embedding。MLP参数在每个worker上都有完整的一份复制, 在参数更新时，会通过allreduce的方式同步不同worker的更新。

### 依赖

- 混合并行使用Horovod做底层的通信, 因此需要安装Horovod, 可以直接使用下面的镜像
- mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:sok-tf212-gpus-v5
  ```
    sudo docker run --gpus=all --privileged -v /home/easyrec/:/home/easyrec/ -ti mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:sok-tf212-gpus-v5 bash
  ```

### 配置

- 修改train_config.train_distribute为EmbeddingParallelStrategy
  ```
   train_config {
      ...
      train_distribute: EmbeddingParallelStrategy
      ...
   }
  ```
- 如使用key-Value Embedding, 需要设置model_config.ev_params
  ```
  model_config {
    ...
    ev_params {
    }
    ...
  }
  ```

### 命令

- 训练
  ```
     CUDA_VISIBLE_DEVICES=0,1,2,4 horovodrun -np 4  python -m easy_rec.python.train_eval --pipeline_config_path samples/model_config/dlrm_on_criteo_parquet_ep_v2.config
  ```
- 评估
  ```
     CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python -m easy_rec.python.eval --pipeline_config_path samples/model_config/dlrm_on_criteo_parquet_ep_v2.config
  ```
  - 注意: 评估目前仅支持单个worker评估
- 导出
  ```
    CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python -m easy_rec.python.export --pipeline_config_path samples/model_config/dlrm_on_criteo_parquet_ep_v2.config --export_dir dlrm_criteo_export/
  ```
