# PAI-NNI-HPO

## GetStarted

注意NNI仅支持python>=3.7,因此请配置python>=3.7的环境

It is tested and supported on Ubuntu >= 18.04, Windows 10 >= 21H2, and macOS >= 11.

### 下载安装easyrec

```bash
git clone https://github.com/alibaba/EasyRec.git
cd EasyRec
bash scripts/init.sh
python setup.py install
```

## 启动调优

### 启动命令

```bash
cd easy_rec/python/hpo_nni/pai_nni/source_begin
nnictl create --config config.yml --port=8780
```

其中port可以是机器上任意未使用的端口号。需要注意的是，NNI实验不会自动退出，如果需要关闭实验请运行nnictl stop主动关闭。
您也可以参考[NNI参考手册](https://nni.readthedocs.io/en/v2.1/Tutorial/QuickStart.html)
查看nnictl的更多用法。

启动成功界面：
![image.png](../../images/automl/pai_nni_create.jpg)

#### config.yml 参数说明

config.yml是作为NNI的配置文件，将代码和搜索空间进行结合，并使用指定的环境来运行您的训练代码，具体参考此config.yml文件。在这里，您还可以还提供其他信息，例如并发度、调优算法、最大Trial数量和最大持续时间等参数。

```
searchSpaceFile: search_space.json
trialCommand: python3 ./run_begin.py --odps_config=../config/odps_config.ini --oss_config=../config/.ossutilconfig --easyrec_cmd_config=../config/easyrec_cmd_config_begin --metric_config=../config/metric_config --exp_dir=../exp
trialConcurrency: 3
maxTrialNumber: 10
tuner:
  name: TPE
  classArgs:
    optimize_mode: maximize
trainingService:
  platform: local
assessor:
   codeDirectory: ../code
   className: pai_assessor.PAIAssessor
   classArgs:
      optimize_mode: maximize
      start_step: 2
```

#### odps账号信息文件

```
project_name=xxx
access_id=xxx
access_key=xxx
end_point=http://service.odps.aliyun.com/api
# MaxCompute 服务的访问链接
tunnel_endpoint=http://dt.odps.aliyun.com
# MaxCompute Tunnel 服务的访问链接
log_view_host=http://logview.odps.aliyun.com
# 当用户执行一个作业后，客户端会返回该作业的 LogView 地址。打开该地址将会看到作业执行的详细信息
https_check=true
#决定是否开启 HTTPS 访问
```

#### oss_config : oss配置文件

```json
[Credentials]
language=ch
endpoint = oss-cn-beijing.aliyuncs.com
accessKeyID = xxx
accessKeySecret= xxx
```

#### easyrec_cmd_config_begin: easyrec命令配置文件

相关参数说明参考[MaxCompute Tutorial](../quick_start/mc_tutorial.md)：

推荐按照以下方式将相关参数以key=value的方式写入config

```
-name=easy_rec_ext
-project=algo_public
-Dscript=oss://easyrec/xxx/software/easy_rec_ext_615_res.tar.gz
-Dtrain_tables=odps://pai_rec_dev/tables/rec_sv_rebuild_acc_rnk_rank_sample_embedding_shuffle_20220529
-Deval_tables=odps://pai_rec_dev/tables/rec_sv_rebuild_acc_rnk_rank_sample_embedding_modify_small/dt=20220530
-Dcmd=train
-Deval_method=separate
-Dconfig=oss://easyrec/xxx/pipeline.config
-Dmodel_dir=oss://easyrec/xxx/deploy/
-Dselected_cols=is_valid_play,ln_play_time,is_like,is_comment,features,content_features
-Dbuckets=oss://easyrec/
-Darn=xxx
-DossHost=oss-cn-beijing-internal.aliyuncs.com
-Dcluster={"ps":{"count":1,"cpu":1600,"memory":40000},"worker":{"count":12,"cpu":1600,"memory":40000}}
```

#### metric_config : 超参数评估方法

多目标示例：metric=val('auc_is_valid_play')\*0.5+val('auc_is_like')\*0.25+val('auc_is_comment')\*0.25

```json
auc_is_valid_play=0.5
auc_is_like=0.25
auc_is_comment=0.25
```

多目标示例：metric=val('auc_is_valid_play')\*0.5+val('auc_is_like')\*0.25+val('auc_is_comment')\*0.25-val('loss_play_time')\*0.25

注意：如果按照metric越大越好的方式优化，loss相关的指标权重定义为负值。

```json
auc_is_valid_play=0.5
auc_is_like=0.25
auc_is_comment=0.25
loss_play_time=-0.25
```

单目标示例：metric=val('auc_is_valid_play')\*1

```json
auc_is_valid_play=1
```

#### 配置超参搜索空间search_space.json

- key是Dconfig中的参数名称，相关配置参考[EasyRecConfig参考手册](../reference.md)
- type是nni中定义的搜索类型，相关配置参考[NNI searchSpace参考手册](https://nni.readthedocs.io/en/v2.2/Tutorial/SearchSpaceSpec.html)
- value是根据业务、经验设置相关搜索值

```json
{
"train_config.optimizer_config[0].adam_optimizer.learning_rate.exponential_decay_learning_rate.initial_learning_rate":{"_type":"choice","_value":[1e-3,1e-4]},
"feature_configs[:].embedding_dim": {"_type": "randint", "_value": [4,16]},
"feature_configs[0].hash_bucket_size": {"_type": "randint", "_value": [1260935, 2521870]},
"model_config.embedding_regularization":{"_type":"uniform","_value":[0.000001, 0.0001]},
}

```

##### key配置注意项

- train_config.optimizer_config\[0\].adam_optimizer.learning_rate.exponential_decay_learning_rate.initial_learning_rate是一个浮点数，注意要用全路径
- feature_configs是一个数组，所以需要用到选择器
  ![image.png](../../images/automl/pai_field.png)
  ```
    - 支持根据属性值选择特征：feature_configs[input_names[0]=field1].embedding_dim，其中input_names[0]=field_name1是选择器
    - 支持使用>=, <=, >, <选择特征，如:feature_configs[inputs_names[0]>=click_]选择名称排在"click_"后面的特征
    - 支持数字作为选择器, 如: feature_configs[0], feature_configs[1]
    - 支持使用:选择所有的特征，如:
        - feature_configs[:]选择全部特征
        - feature_configs[5:]选择index从5开始的特征
        - feature_configs[:13]选择index从0到12的特征
        - feature_configs[3:12]选择index从3到11的特征

  ```
- model_config.embedding_regularization 是一个浮点数，注意使用全路径

##### type配置注意事项

[NNI searchSpace参考手册](https://nni.readthedocs.io/en/v2.2/Tutorial/SearchSpaceSpec.html)

- {"\_type": "choice", "\_value": options}：从options中选取一个。
- {"\_type": "randint", "\_value": \[lower, upper\]}：\[low,upper)之间选择一个随机整数。
- {"\_type": "uniform", "\_value": \[low, high\]}：\[low,upper\]之间随机采样。

## 调优结果

在运行实验后，可以在命令行界面中找到如下的Web界面地址 ：\[Your IP\]:\[Your Port\]
![image.png](../../images/automl/pai_nni_create.jpg)

### 查看概要页面

在这里可以看到实验相关信息，如配置文件、搜索空间、运行时长、日志路径等。NNI 还支持通过 Experiment summary 按钮下载这些信息和参数。
![image.png](../../images/automl/pai_nni_overview.jpg)

### 查看Trial详情页面

您可以在此页面中看到整个实验过程中，每个trial的结果情况。
![image.png](../../images/automl/pai_nni_detail.jpg)

## finetune训练（可选）

由于推荐业务每天都有实时更新的数据，如果用户采用先训练一批历史数据，后面每天finetune更新模型的话，可以利用以上begin调优的最优结果，再在新数据上微调。如果用户每次更新模型都是重新开始训练的话，则不需要此步骤。

### 调优经验

例如：用户有40天历史数据，可以先利用以上步骤调优30天数据，然后根据搜索出的最优参数，再finetuen剩余10天。
经验是：根据begin训练得出的最优参数，将learning_rate设置为begin结束时的learning_rate。
例如：
begin训练时learning_rate如下,begin训练总计为8000步，因此可以设置finetune时initial_learning_rate=1e-6或者1e-7：

```
learning_rate {
        exponential_decay_learning_rate {
          initial_learning_rate: 0.001
          decay_steps: 1000
          decay_factor: 0.1
          min_learning_rate: 1e-07
        }
      }
```

支持手动修改，也支持代码修改配置，修改效果如下：
![image.png](../../images/automl/modify_lr.jpg)

#### 使用代码修改配置(可选)

支持本地上pipeline文件修改

```bash
cd easy_rec/python/hpo_nni/pai_nni/code
python modify_pipeline_config.py --pipeline_config_path=../config/pipeline.config --save_path=../config/pipeline_finetune.config --learning_rate=1e-6
```

也支持oss上pipeline文件直接修改

```bash
cd easy_rec/python/hpo_nni/pai_nni/code
python modify_pipeline_config.py --pipeline_config_path=oss://easyrec/yj374186/pipeline889.config --save_path=oss://easyrec/yj374186/pipeline889-f.config --learning_rate=1e-6 --oss_config=../config/.ossutilconfig
```

如果用户想要看是否有更优参数，可以看下级目录启动调优。

### 启动调优(可选)

```bash
cd source_finetune
nnictl create --config config.yml --port=8617
```

#### config.yml

```
searchSpaceFile: search_space.json
trialCommand: python3 ./run_finetune.py --odps_config=../config/odps_config.ini --oss_config=../config/.ossutilconfig --easyrec_cmd_config=../config/easyrec_cmd_config_finetune --metric_config=../config/metric_config --exp_dir=../exp --start_time=2022-05-30 --end_time=2022-06-01
trialConcurrency: 3
maxTrialNumber: 10
tuner:
  name: TPE
  classArgs:
    optimize_mode: maximize
trainingService:
  platform: local
assessor:
   codeDirectory: ../code
   className: pai_assessor.PAIAssessor
   classArgs:
      optimize_mode: maximize
      start_step: 2
```

#### easyrec_cmd_config_finetune: easyrec命令配置文件

相关参数说明参考[MaxCompute Tutorial](../quick_start/mc_tutorial.md)：

```
-name=easy_rec_ext
-project=algo_public
-Dscript=oss://easyrec/xxx/software/easy_rec_ext_615_res.tar.gz
-Dtrain_tables=odps://pai_rec_dev/tables/rec_sv_rebuild_acc_rnk_rank_sample_embedding_shuffle_{bizdate}
-Deval_tables=odps://pai_rec_dev/tables/rec_sv_rebuild_acc_rnk_rank_sample_embedding_modify/dt={eval_ymd}
-Dcmd=train
-Deval_method=separate
-Dconfig=oss://easyrec/xxx/pipeline_finetune.config
-Dedit_config_json={"train_config.fine_tune_checkpoint":"oss://easyrec/xxx/finetune/{predate}_finetune_model_nni_622"}
-Dmodel_dir=oss://easyrec/xxx/finetune/{bizdate}_finetune_model_nni_622
-Dselected_cols=is_valid_play,ln_play_time,is_like,is_comment,features,content_features
-Dbuckets=oss://easyrec/
-Darn=xxx
-DossHost=oss-cn-beijing-internal.aliyuncs.com
-Dcluster={"ps":{"count":1,"cpu":1600,"memory":40000},"worker":{"count":12,"cpu":1600,"memory":40000}}
```

与begin训练的`差异点`:

- Dedit_config_json: 需要配置一下train_config.fine_tune_checkpoint的路径,且需要和Dmodel_dir对应
- 假设每天finetune：
  - {bizdate} 必须保留，将会在代码中根据当天日期进行替换
  - {eval_ymd} 必须保留，将会在代码中根据第二天日期进行替换
  - {predate} 必须保留，将会在代码中根据前一天日期进行替换

#### run.py 参数说明

表示从20220530finetune到20220617，根据这些天的平均结果衡量超参数的优劣。

```
parser.add_argument(
      '--start_time', type=str, help='finetune start time', default='2022-05-30')
parser.add_argument(
    '--end_time', type=str, help='finetune end time', default='2022-06-17')
```

#### 配置超参搜索空间search_space.json

参考begin训练阶段中想要搜索的参数即可，注意由于是finetune训练，网络结构相关的参数不要进行搜索，经验是搜索LR

```
learning_rate {
        exponential_decay_learning_rate {
          initial_learning_rate: 0.001
          decay_steps: 1000
          decay_factor: 0.1
          min_learning_rate: 1e-07
        }
      }
```

finetune时可以沿用以上的learning_rate策略，搜索一下初始化LR，注意finetuning时LR需要缩小。

search_space.json：

```
{
"train_config.optimizer_config[0].adam_optimizer.learning_rate.exponential_decay_learning_rate.initial_learning_rate":{"_type":"choice","_value":[1e-6,1e-5,1e-4]},
}
```

## 修改代码（可选）

如果您想设置自定义停止策略，可以参考[NNI CustomizeAssessor](https://nni.readthedocs.io/en/v2.6/Assessor/CustomizeAssessor.html)

注意继承pai_nni/code/pai_custom_assessor.PaiCustomizedAssessor
trial_end函数，该函数是用来当一个实验被停止时，会去将maxcompute作业给终止掉。

```
def trial_end(self, trial_job_id, success):
        print("trial end")
        if not success:
            print("early stop kill instance")
            access_id = get_value('access_id',trial_id=trial_job_id)
            access_key = get_value('access_key',trial_id=trial_job_id)
            project = get_value('project',trial_id=trial_job_id)
            endpoint = get_value('endpoint',trial_id=trial_job_id)
            instance = get_value(trial_job_id,trial_id=trial_job_id)
            if access_id and access_key and project and endpoint and instance:
                o = create_odps(access_id=access_id, access_key=access_key, project=project, endpoint=endpoint)
                print("stop instance")
                o.stop_instance(instance)
                print("stop instance success")
                # for report result
                set_value(trial_job_id+'_exit','1',trial_id=trial_job_id)
```
