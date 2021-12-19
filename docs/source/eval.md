# 评估

### eval_config

```sql
eval_config {
  metrics_set: {
    # metric为auc
    auc {}
  }
}
```

- metrics_set: 配置评估指标，可以配置多个，如:

```sql
eval_config {
  metrics_set: {
    # metric为auc
    auc {}
    # metric为mae
    mean_absolute_error {}
  }
}
```

- num_examples: 默认为0, 表示评估所有样本；大于0，则每次只评估num_examples样本，一般在调试或者示例的时候使用

### Metric:

| MetricClass        | Example                   | 适用模型                                                                  |
| ------------------ | ------------------------- | --------------------------------------------------------------------- |
| Accuracy           | accuracy {}               | 多分类模型LossType=CLASSIFICATION, num_class > 1                           |
| MeanAbsoluteError  | mean_absolute_error {}    | 回归模型LossType=L2_LOSS                                                  |
| RecallAtTopK       | recall_at_topk {}         | 多分类模型LossType=CLASSIFICATION, num_class > 1; CoMetricLearningI2I模型    |
| Max_F1             | max_f1 {}                 | 分类模型LossType=CLASSIFICATION                                           |
| MeanSquaredError   | mean_squared_error{}      | 回归模型LossType=L2_LOSS                                                  |
| AUC                | auc{}                     | 二分类模型LossType=CLASSIFICATION, num_class = 1                           |
| GAUC               | gauc{}                    | 二分类模型LossType=CLASSIFICATION, num_class = 1                           |
| SessionAUC         | session_auc{}             | 二分类模型LossType=CLASSIFICATION, num_class = 1                           |
| Precision          | precision{}               | 二分类模型LossType=CLASSIFICATION, num_class = 1                           |
| Recall             | recall{}                  | 二分类模型LossType=CLASSIFICATION, num_class = 1                           |
| AvgPrecisionAtTopK | precision_at_topk{topk=5} | CoMetricLearningI2I模型专用, LossType=CIRCLE_LOSS / MULTI_SIMILARITY_LOSS |

### 评估命令

#### Local

```bash
python -m easy_rec.python.eval --pipeline_config_path dwd_avazu_ctr_deepmodel.config
```

- --pipeline_config_path: config文件路径
- --model_dir: 如果指定了model_dir将会覆盖config里面的model_dir，一般在周期性调度的时候使用

#### PAI

```sql
pai -name easy_rec_ext -project algo_public
-Dcmd=evaluate
-Dconfig=oss://easyrec/config/MultiTower/dwd_avazu_ctr_deepmodel_ext.config
-Dtables=odps://pai_online_project/tables/dwd_avazu_ctr_deepmodel_test
-Dcluster='{"worker" : {"count":1, "cpu":1000, "gpu":100, "memory":40000}}'
-Dmodel_dir=oss://easyrec/ckpt/MultiTower
-Darn=acs:ram::xxx:role/xxx
-Dbuckets=oss://easyrec/
-DossHost=oss-cn-beijing-internal.aliyuncs.com；
```

- -Dcmd: evaluate 模型评估
- -Dconfig: 同训练
- -Dtables: 只需要指定测试 tables
- -Dcluster: 评估不需要PS节点，指定一个worker节点即可
- -Dmodel_dir: 如果指定了model_dir将会覆盖config里面的model_dir，一般在周期性调度的时候使用
- -Dcheckpoint_path: 使用指定的checkpoint_path，如oss://easyrec/ckpt/MultiTower/model.ckpt-1000。不指定的话，默认model_dir中最新的ckpt文件。
- 如果是pai内部版,则不需要指定arn和ossHost, arn和ossHost放在-Dbuckets里面
  - -Dbuckets=oss://easyrec/?role_arn=acs:ram::xxx:role/ev-ext-test-oss&host=oss-cn-beijing-internal.aliyuncs.com

#### 分布式评估

```sql
pai -name easy_rec_ext -project algo_public
-Dcmd=evaluate
-Dconfig=oss://easyrec/config/MultiTower/dwd_avazu_ctr_deepmodel_ext.config
-Dtables=odps://pai_online_project/tables/dwd_avazu_ctr_deepmodel_test
-Dcluster='{"worker" : {"count":1, "cpu":1000, "gpu":100, "memory":40000}}'
-Dmodel_dir=oss://easyrec/ckpt/MultiTower
-Dextra_params=" --distribute_eval True"
-Darn=acs:ram::xxx:role/xxx
-Dbuckets=oss://easyrec/
-DossHost=oss-cn-beijing-internal.aliyuncs.com；
```

- -distribute_eval: 分布式 evaluate

评估的结果会写到model_dir目录下的文件"eval_result.txt"中。
