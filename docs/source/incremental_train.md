# 增量训练

增量训练的优势:

- 每当有新增的样本时，并不需要重新训练全量样本，而是用现有模型初始化，然后在当天的新增样本上继续finetune。实验表明，增量训练可以使模型迅速收敛，加快训练速度，并且由于见过了更多的样本，所以模型性能也有一定的提升。增量训练在新数据上finetune一个epoch即可达到比较好的效果。

### 训练命令

#### Local

初始化：

```bash
python -m easy_rec.python.train_eval --pipeline_config_path dwd_avazu_ctr_deepmodel.config
```

增量训练：

```bash
python -m easy_rec.python.train_eval --pipeline_config_path dwd_avazu_ctr_deepmodel.config --edit_config_json='{"train_config.fine_tune_checkpoint": "${bizdate-1}/model.ckpt-50", "train_config.num_steps": 10000}'
```

- bizdate是业务日期，一般是运行日期-1day.

#### on PAI

初始化：

```sql
pai -name easy_rec_ext -project algo_public
-Dconfig=oss://easyrec/easy_rec_test/dwd_avazu_ctr_deepmodel_ext.config
-Dcmd=train
-Dtrain_tables=odps://pai_online_project/tables/train_data_d1_to_d14
-Deval_tables=odps://pai_online_project/tables/eval_data/ds=${bizdate}
-Dmodel_dir="oss://easyrec/easy_rec_test/checkpoints/${bizdate}/"
-Dcluster='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":3, "cpu":1000, "gpu":100, "memory":40000}}'
-Darn=acs:ram::xxx:role/ev-ext-test-oss
-Dbuckets=oss://easyrec/
-DossHost=oss-cn-beijing-internal.aliyuncs.com
-Dwith_evaluator=1;
```

增量训练：

```sql
pai -name easy_rec_ext -project algo_public
-Dconfig=oss://easyrec/easy_rec_test/dwd_avazu_ctr_deepmodel_ext.config
-Dcmd=train
-Dtrain_tables=odps://pai_online_project/tables/train_data/ds=${bizdate}
-Deval_tables=odps://pai_online_project/tables/eval_data/ds=${bizdate}
-Dcluster='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":3, "cpu":1000, "gpu":100, "memory":40000}}'
-Darn=acs:ram::xxx:role/ev-ext-test-oss
-Dbuckets=oss://easyrec/
-DossHost=oss-cn-beijing-internal.aliyuncs.com
-Dedit_config_json='{"train_config.fine_tune_checkpoint": "oss://easyrec/easy_rec_test/checkpoints/${bizdate-1}/"}'
-Dwith_evaluator=1;
```

- bizdate在dataworks里面是业务日期，一般是运行日期的前一天。
- train_config.fine_tune_checkpoint:  fine_tune_checkpoint的路径，可以指定具体的checkpoint，也可以指定一个目录，将自动定位目录里面最新的checkpoint。
