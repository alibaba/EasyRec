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
python modify_pipeline_config.py --pipeline_config_path=./samples/pipeline.config --save_path=./samples/pipeline_finetune.config --learning_rate=1e-6
```

也支持oss上pipeline文件直接修改

```bash
python modify_pipeline_config.py  --pipeline_config_path=oss://easyrec/pipeline889.config --save_path=oss://easyrec/pipeline889-f.config --learning_rate=1e-6 --oss_config=../config/.ossutilconfig
```

如果用户想要看是否有更优参数，可以看下级目录启动调优。

### 启动调优(可选)

```bash
nnictl create --config config_finetune.yml --port=8617
```

#### config_finetune.ini

```
[platform_config]
name=MaxCompute
{% set date_list = [20220616,20220617] %}
{% set date_begin = 20220616 %}
{% for bizdate in date_list %}
{% set eval_ymd = bizdate +1 %}
{% set predate = bizdate -1 %}
{% if bizdate == date_begin %}
cmd1_{{bizdate}}=PAI -name=easy_rec_ext
    -project=algo_public
    -Dscript='oss://automl-nni/easyrec/easy_rec_ext_615_res.tar.gz'
    -Dtrain_tables='odps://pai_rec_dev/tables/rec_sv_rebuild_acc_rnk_rank_sample_embedding_modify/dt={{bizdate}}'
    -Deval_tables='odps://pai_rec_dev/tables/rec_sv_rebuild_acc_rnk_rank_sample_embedding_modify/dt={{eval_ymd}}'
    -Dcmd=train
    -Deval_method=separate
    -Dfine_tune_checkpoint="oss://automl-nni/easyrec/finetune/{{predate}}_finetune_model_nni_622"
    -Dconfig='oss://automl-nni/easyrec/config/easyrec_model_${exp_id}_${trial_id}.config'
    -Dmodel_dir='oss://automl-nni/easyrec/finetune/{{bizdate}}_finetune_model_nni_622/${exp_id}_${trial_id}'
    -Dselected_cols='is_valid_play,ln_play_time,is_like,is_comment,features,content_features'
    -Dbuckets='oss://automl-nni/'
    -Darn='xxx'
    -DossHost='oss-cn-beijing-internal.aliyuncs.com'
    -Dcluster={"ps":{"count":1,"cpu":1600,"memory":40000 },"worker":{"count":12,"cpu":1600,"memory":40000}}

{% else %}
cmd1_{{bizdate}}=PAI -name=easy_rec_ext
    -project=algo_public
    -Dscript='oss://automl-nni/easyrec/easy_rec_ext_615_res.tar.gz'
    -Dtrain_tables='odps://pai_rec_dev/tables/rec_sv_rebuild_acc_rnk_rank_sample_embedding_modify/dt={{bizdate}}'
    -Deval_tables='odps://pai_rec_dev/tables/rec_sv_rebuild_acc_rnk_rank_sample_embedding_modify/dt={{eval_ymd}}'
    -Dcmd=train
    -Deval_method=separate
    -Dfine_tune_checkpoint="oss://automl-nni/easyrec/finetune/{{predate}}_finetune_model_nni_622/${exp_id}_${trial_id}"
    -Dconfig='oss://automl-nni/easyrec/config/easyrec_model_${exp_id}_${trial_id}.config'
    -Dmodel_dir='oss://automl-nni/easyrec/finetune/{{bizdate}}_finetune_model_nni_622/${exp_id}_${trial_id}'
    -Dselected_cols='is_valid_play,ln_play_time,is_like,is_comment,features,content_features'
    -Dbuckets='oss://automl-nni/'
    -Darn='xxx'
    -DossHost='oss-cn-beijing-internal.aliyuncs.com'
    -Dcluster={"ps":{"count":1,"cpu":1600,"memory":40000 },"worker":{"count":12,"cpu":1600,"memory":40000}}
{% endif %}

{% endfor %}


[metric_config]
# metric type is summary/table
metric_type=summary
{% set date_list = [20220616,20220617] %}
{% for bizdate in date_list %}
metric_source_{{bizdate}}=oss://automl-nni/easyrec/finetune/{{bizdate}}_finetune_model_nni_622/${exp_id}_${trial_id}/eval_val/
{% endfor %}
# best/final/avg,default=best
final_mode=final
source_list_final_mode=avg
metric_dict={'auc_is_like':0.25, 'auc_is_valid_play':0.5, 'auc_is_comment':0.25}
```

与begin训练的`差异点`:

- 每个配置模块支持jinja模版渲染
- 配置finetune日期{% set date_list = \[20220616,20220617\] %}
- 配置finetune开始日期{% set date_begin = 20220616 %}，Dfine_tune_checkpoint开始日期和后续日期采取的model路径不一样
- 假设每天finetune：
  - {bizdate} 必须保留，将会在代码中根据当天日期进行替换
  - {eval_ymd} 必须保留，将会在代码中根据第二天日期进行替换
  - {predate} 必须保留，将会在代码中根据前一天日期进行替换
- metric_source也是多条路径，每一天训练结果为summary的最终结果，整组参数finetune的结果为这些天的平均值

#### 配置超参搜索空间search_space.json

参考begin训练阶段中想要搜索的参数即可，注意由于是finetune训练，网络结构相关的参数不要进行搜索，经验是搜索LR
