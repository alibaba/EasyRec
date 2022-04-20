# 预检查

为解决用户常由于脏数据或配置错误的原因，导致训练失败，开发了预检查功能。
在训练时打开检查模式，或是训练前执行pre_check脚本，即会检查data_config配置及train_config部分配置，筛查全部数据，遇到异常则抛出相关信息，并给出修改意见。


### 命令

#### Local

方式一: 执行pre_check脚本：
```bash
PYTHONPATH=. python easy_rec/python/tools/pre_check.py --pipeline_config_path samples/model_config/din_on_taobao.config --data_input_path data/test/check_data/csv_data_for_check
```

方式二: 训练时打开检查模式（默认关闭）：

该方式会影响训练速度，线上例行训练时不建议开启检查模式。
```bash
python -m easy_rec.python.train_eval --pipeline_config_path samples/model_config/din_on_taobao.config --check_mode
```
- pipeline_config_path config文件路径
- data_input_path 待检查的数据路径，不指定的话为pipeline_config_path中的train_input_path及eval_input_path
- check_mode 默认False


#### On PAI

方式一: 执行pre_check脚本：
```sql
pai -name easy_rec_ext -project algo_public
  -Dcmd='check'
  -Dconfig='oss://easyrec/easy_rec_test/dwd_avazu_ctr_deepmodel_ext.config'
  -Dtables='odps://pai_online_project/tables/dwd_avazu_ctr_deepmodel_train,odps://pai_online_project/tables/dwd_avazu_ctr_deepmodel_test'
  -Dselected_cols='isclick,label,view_costtime,getgiftnum,is_on_wheat,on_wheat_duration,features'
  -Darn=acs:ram::xxx:role/ev-ext-test-oss
  -Dbuckets=oss://easyrec/
  -DossHost=oss-cn-beijing-internal.aliyuncs.com
  -Dcluster='{\"worker\":{\"count\":3,\"cpu\":800}}';
```

方式二: 训练时打开检查模式（默认关闭）：

该方式会影响训练速度，线上例行训练时不建议开启检查模式。
```sql
pai -name easy_rec_ext  -project algo_public
  -Dcmd='train'
  -Dconfig='oss://easyrec/easy_rec_test/dwd_avazu_ctr_deepmodel_ext.config'
  -Dtrain_tables='odps://pai_online_project/tables/dwd_avazu_ctr_deepmodel_train'
  -Deval_tables='odps://pai_online_project/tables/dwd_avazu_ctr_deepmodel_test'
  -Dselected_cols='isclick,label,view_costtime,getgiftnum,is_on_wheat,on_wheat_duration,features'
  -Dmodel_dir='oss://easyrec/easy_rec_test/dwd_avazu_ctr_deepmodel_ext/ckpt'
  -Dextra_params='--check_mode'
  -Darn=acs:ram::xxx:role/ev-ext-test-oss
  -Dbuckets=oss://easyrec/
  -DossHost=oss-cn-beijing-internal.aliyuncs.com
  -Dcluster='{
      \"ps\": {
          \"count\" : 1,
          \"cpu\" : 1600
      },
      \"worker\" : {
          \"count\" : 3,
          \"cpu\" : 800
      }
  }';
```

- -Dtables: 待检查的表路径，可以指定多个，逗号分隔
- -Dtrain_tables: 训练表，可以指定多个，逗号分隔
- -Deval_tables: 评估表，可以指定多个，逗号分隔
- -Dcluster: 定义ps和worker的配置，方式一无需指定ps节点
- -Dconfig: config文件路径
- -Darn: rolearn  注意这个的arn要替换成客户自己的。可以从dataworks的设置中查看arn。
- -DossHost: ossHost地址
- -Dbuckets: config所在的bucket和保存模型的bucket; 如果有多个bucket，逗号分割
