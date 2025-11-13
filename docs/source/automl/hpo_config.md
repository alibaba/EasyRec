HPO启动配置包含exp.yml. trial.ini, search_space.json三个模块。

# exp.yml

exp.yml是作为NNI的配置文件，将代码和搜索空间进行结合，并使用指定的环境来运行您的训练代码，具体参考此exp.yml文件。在这里，您还可以还提供其他信息，例如并发度、调优算法、最大Trial数量和最大持续时间等参数。https://nni.readthedocs.io/zh/stable/reference/experiment_config.html#experimentconfig

## 字段内容

字段可以直接参考NNI官网，区别在于为了结合PAI,这些字段需保持不变

```
trialCommand: python3 -m hpo_tools.core.utils.run --config=./trial.ini
trainingService:
  platform: local
assessor:
  name: PAIAssessor
```

同时，为了能够停止PAI任务，需要使用PAIAssessor

## PAIAssessor

```
支持将该组中的实验结果和同组中的所有历史进行比较，如果不满足比较标准（例如小于中位数），则停止该组超参数的运行。比如说设置最大运行次数max_trial_num， 实际使用量会显著小于max_trial_num，但具体数量就和实际跑的任务及随机到的超参有关系了。例如max_trial_num=50时，可能最终可能不到 25 次，并且差不多已经是完整探索了50组超参。
```

| PAIAssessor   | 描述                            | 值                 |
| ------------- | ----------------------------- | ----------------- |
| optimize_mode | 最大化优化的方向                      | maximize/minimize |
| start_step    | 从第几步开始进行早停判定                  | 2                 |
| moving_avg    | 早停判断时，采用所有历史的滑动平均值作为判断标准      | True              |
| proportion    | 本次超参搜索的最优值和历史记录的proportion值比较 | 0.5               |
| patience      | metric指标连续下降几次，就停止            | 10                |

### 示例

```
experimentWorkingDirectory: ../expdir
searchSpaceFile: search_space.json
trialCommand: python3 -m hpo_tools.core.utils.run --config=./trial.ini
trialConcurrency: 1
maxTrialNumber: 4
tuner:
  name: TPE
  classArgs:
    optimize_mode: maximize
debug: true
logLevel: debug
trainingService:
  platform: local
assessor:
  name: PAIAssessor
  classArgs:
    platform: MAXCOMPUTE
    optimize_mode: maximize
    start_step: 1
    moving_avg: true
    proportion: 0.5
```

# trial.ini

## 变量替换原则

### 值替换

程序会将trial.ini 中以下这些key默认替换成对应的值。参数默认支持值替换、列表替换、字典替换、json替换、文件替换（params_config)、支持嵌套字典的key替换（组合参数例子dlc_mnist_nested_search_space)

- cmd = cmd.replace('${exp_id}', experment_id.lower())

- cmd = cmd.replace('${trial_id}', trial_id.lower())

- cmd = cmd.replace('${NNI_OUTPUT_DIR}',os.environ.get('NNI_OUTPUT_DIR', './tmp'))

- cmd = cmd.replace('${tuner_params_list}', tuner_params_list)

- cmd = cmd.replace('${tuner_params_dict}', tuner_params_dict)

- cmd = cmd.replace('${tuner_params_json}', json.dumps(tuner_params))

- cmd = cmd.replace('${params}', params)->支持参数标识路径，例如lr0.001_batchsize64 注意其中可能含有浮点数，请确定是否支持用来标识数据/数据表

- cmd = cmd.replace(p, str(v)) 将搜索的参数替换为搜索的值，搜索参数可以使用${batch_size}、${lr}来标记，需要和search_space.json中的key匹配使用

### jinja渲染

每个配置模块支持jinja模版渲染，用于用户在一开始设置变量，具体可以查看案例cross-validation/maxcompute-easyrec

```
[metric_config]
# metric type is summary/table
metric_type=summary
{% set date_list = [20220616,20220617] %}
{% for bizdate in date_list %}
metric_source_{{bizdate}}=oss://automl-nni/easyrec/finetune/{{bizdate}}_finetune_model_nni_622/${exp_id}_${trial_id}/eval_val/
{% endfor %}
```

### 字段介绍

| 配置模块            | 描述                                                                | 是否可选 |
| --------------- | ----------------------------------------------------------------- | ---- |
| platform_config | 用于标记任务执行的平台以及对应的执行命令                                              | 必选   |
| metric_config   | 用于标记任务metric的获取来源、metric的key以及对应权重、metric类型、最终metric的方式           | 必选   |
| output_config   | 如果使用服务版，可以配置output_config用来获取最优模型配置summary_path，用于配制tensorboard路径 | 可选   |
| schedule_config | 如果任务在指定时间内调度任务，则需要配置schedule_config,修改对应的schedule_config的值        | 可选   |
| params_config   | 如果用户的参数是保存在文件中，则需要配置params_config, 用于标记需要修改参数的源文件路径和目标路径          | 可选   |
| oss_config      | 如果任务需要使用OSS存储，则需要配置OSS config                                     | 可选   |
| odps_config     | 如果任务需要使用maxcompute平台执行任务，则需要配置odps config                         | 可选   |
| ts_config       | 如果任务需要使用trainingservice平台执行任务，则需要配置ts config                      | 可选   |
| paiflow_config  | 如果任务需要执行工作流任务，则需要配置paiflow_config,修改对应的paiflow_config的值           | 可选   |
| dlc_config      | 如果任务需要执行dlc任务，则需要配置dlc_config,修改对应的dlc_config的值                   | 可选   |
| monitor_config  | 支持失败告警,最优metric更新时提醒                                              | 可选   |

## platform_config

| platform_config | 描述                                                           | 值                                                              |
| --------------- | ------------------------------------------------------------ | -------------------------------------------------------------- |
| name            | 用于标记任务执行的平台                                                  | DLC/MaxCompute/DataScience/LOCAL/PAI/PAIFLOW                   |
| cmdxx           | 用于标记执行的命令，以cmd开头                                             | dlc submit pytorch --name=test_nni\_${exp_id}\_${trial_id} xxx |
| resume          | 1表示开启续跑模式；用于用户一次运行时，比如说第一行任务成功，第二行由于资源不足失败，可以开启续跑，从第二行命令开始运行 | 0/1                                                            |

## metric_config

| metric_config          | 描述                                                                                                     | 值                                                                                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| metric_type            | metric类型                                                                                               | summary/table/api/json/stdout                                                                                                                                                                                                  |
| metric_source          | metric来源（可以为多个以metric_source开头的，具体可以看maxcompute_crossvalidation案例）                                     | 对应为具体的路径或者job                                                                                                                                                                                                                  |
| final_mode             | 如果任务运行过程中，存在很多中间metric，那么需要确定最终metric的计算方式                                                             | final/best/avg                                                                                                                                                                                                                 |
| source_list_final_mode | 可选，默认值为final_mode，可选值为final/best/avg,用于有多个metric_source时最终metric如何计算，具体可以看maxcompute_crossvalidation案例 | final/best/avg                                                                                                                                                                                                                 |
| metric_dict            | 对应查询的key以及对应的权重;可以为负值                                                                                  | metric_dict={'auc_is_like':0.25, 'auc_is_valid_play':0.5, 'auc_is_comment':0.25, 'loss_play_time':-0.25} metric=val(’auc_is_valid_play’)\*0.5+val(’auc_is_like’)\*0.25+val(’auc_is_comment’)\*0.25-val(’loss_play_time’)\*0.25 |

- 如果metric_type=stdout类型，则metric_dict对应的key为正则表达式，value为对应的权重

```
[metric_config]
# metric type is summary/table
metric_type=stdout
metric_source=oss://test-nni/examples/search/pai/stdout/stdout_${exp_id}_${trial_id}
# best or final,default=best
final_mode=best
metric_dict={'validation: accuracy=([0-9\\.]+)':1}
```

- 如果metric_type=stdout类型，则metric_source支持指定默认任务的日志为来源
  - stdoutmetric：支持指定具体的任务；例如metric_source=cmd1,即使用cmd1输出的任务日志做正则
  - stdoutmetric：支持指定具体的任务，并过滤文件，例如metric_source=cmd,worker;即使用cmd1任务中所有的worker日志做正则

```
[metric_config]
# metric type is summary/table
metric_type=stdout
# default is cmd, cmd->platform job 1,we will get the job1 all default stdout
# if the job is distributed, you can use [cmd,worker] to assign which log has metric or just use [cmd] to choose all stdout
metric_source=cmd,worker
# best or final,default=best
final_mode=best
metric_dict={'validation: accuracy=([0-9\\.]+)':1}
optimize_mode=maximize
```

- 如果metric_type=summary类型，则metric_source为对应的summary路径

```
[metric_config]
# metric type is summary/table
metric_type=summary
# the easyrec model_dir/eval_val/ have the events summary file
metric_source=hdfs://123.57.44.211:9000/user/nni/datascience_easyrec/model_nni/${exp_id}_${trial_id}/eval_val/
```

- 如果metric_type=table类型，则metric_source为对应的sql语句

```
[metric_config]
# metric type is summary/table
metric_type=table
metric_source=select * from ps_smart_classification_metrics where pt='${exp_id}_${trial_id}';
```

- 如果metric_type=api类型，则metric支持指定具体的任务；例如metric_source=cmd1

```
[metric_config]
# metric type is summary/table/api
metric_type=api
# default is cmd1,cmd1->platform job 1, we will get the default job1 metric
# if is list,metric_source_1=cm1,metric_source_2=cmd2
metric_source=cmd1
```

## output_config

| output_config | 描述                                   | 值   |
| ------------- | ------------------------------------ | --- |
| model_path    | 如果使用服务版，可以配置model_path用来获取最优模型       | 路径  |
| summary_path  | 如果使用单机版，可以配置summary用于本地查看TensorBoard | 路径  |

## schedule_config

| schedule_config | 描述                   | 值                |
| --------------- | -------------------- | ---------------- |
| day             | 支持在指定时间范围内调度AutoML任务 | everyday/weekend |
| start_time      | 指定调度开始时间             | 00:00-23:59      |
| end_time        | 指定调度结束时间             | 00:00-23:59      |

## params_config

如果用户的参数是保存在文件中，则需要配置params_config,

| params_config              | 描述                                                     | 值                 |
| -------------------------- | ------------------------------------------------------ | ----------------- |
| params_src_dst_filepath1xx | 用于标记需要修改参数的源文件路径和目标路径,可以为多个，以params_src_dst_filepath开头 | src_path,dst_path |
| params_src_dst_filepath2xx | xx                                                     | xx                |

## oss_config

| oss_config      | 描述       | 值                                                                          |
| --------------- | -------- | -------------------------------------------------------------------------- |
| endpoint        | endpoint | [http://oss-cn-shanghai.aliyuncs.com](http://oss-cn-shanghai.aliyuncs.com) |
| accessKeyID     | ak       | ak                                                                         |
| accessKeySecret | sk       | sk                                                                         |
| role_arn        | role_arn | acs:ram::xxx:role/aliyunserviceroleforpaiautoml                            |

## odps_config

| odps_config   | 描述           | 值                                                                                     |
| ------------- | ------------ | ------------------------------------------------------------------------------------- |
| access_id     | ak           | ak                                                                                    |
| access_key    | sk           | ak                                                                                    |
| project_name  | project_name | xxx                                                                                   |
| end_point     | end_point    | 弹外: http://service.odps.aliyun.com/api 弹内：http://service-corp.odps.aliyun-inc.com/api |
| log_view_host | logview host | 弹外：http://logview.odps.aliyun.com 弹内：http://logview.alibaba-inc.com                   |
| role_arn      | role_arn     | acs:ram::xxx:role/aliyunserviceroleforpaiautoml                                       |

## dlc_config

| dlc_config | 描述          | 值                                                                 |
| ---------- | ----------- | ----------------------------------------------------------------- |
| access_id  | ak          | ak                                                                |
| access_key | sk          | ak                                                                |
| end_point  | end_point   | 弹外：pai-dlc.cn-shanghai.aliyuncs.com 弹内：pai-dlc-share.aliyuncs.com |
| region     | cn-shanghai | cn-shanghai                                                       |
| protocol   | protocol    | http/https                                                        |

## ts_config

| ts_config         | 描述        | 值                            |
| ----------------- | --------- | ---------------------------- |
| access_key_id     | ak        | ak                           |
| access_key_secret | sk        | ak                           |
| region_id         | reigin    | xxx                          |
| endpoint          | end_point | pai.cn-hangzhou.aliyuncs.com |

## paiflow_config

| paiflow_config    | 描述           | 值       |
| ----------------- | ------------ | ------- |
| access_key_id     | ak           | ak      |
| access_key_secret | sk           | ak      |
| region_id         | reigin       | xxx     |
| workspace_id      | workspace_id | 2332411 |

## monitor_config

- 参考[阿里钉机器人](https://open.dingtalk.com/document/robots/custom-robot-access)去添加自定义机器人，获取url
  - 点击阿里钉头像->机器人管理-自定义机器人->群组选择工作通知
  - 点击阿里钉头像->机器人管理-自定义机器人->群组：选择对应的群号

| monitor_config | 描述                                           | 值                                                     |
| -------------- | -------------------------------------------- | ----------------------------------------------------- |
| url            | url为创建自定义机器人对应的Webhook地址                     | https://oapi.dingtalk.com/robot/send?access_token=xxx |
| keyword        | 添加自定义机器人：自定义关键词                              | monitor                                               |
| at_mobiles     | 在content里添加@人的手机号，且只有在群内的成员才可被@，非群内成员手机号会被脱敏 | \['11xx'\]                                            |
| at_user_ids    | 被@人的用户userid。即工号                             | \[\]                                                  |
| is_at_all      | 是否@所有人                                       | True/False                                            |

## search_space.json

| search_space | 描述                                                                                                        | 值   |
| ------------ | --------------------------------------------------------------------------------------------------------- | --- |
| key          | trial.ini中配置的搜索参数变量                                                                                       |     |
| type         | nni中定义的搜索类型，相关配置参考[NNI searchSpace参考手册](https://nni.readthedocs.io/en/v2.2/Tutorial/SearchSpaceSpec.html) |     |

- {”\_type”: “choice”, “\_value”: options}：从options中选取一个。
- {”\_type”: “randint”, “\_value”: \[lower, upper\]}：\[low,upper)之间选择一个随机整数。
- {”\_type”: “uniform”, “\_value”: \[low, high\]}：\[low,upper\]之间随机采样。
  |
  | value | value是根据业务、经验设置相关搜索值 |
  |

### 示例

```
{
    "${batch_size}": {"_type":"choice", "_value": [16, 32, 64, 128]},
    "${lr}":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]}
}
```
