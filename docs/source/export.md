# 导出

### export_config

```protobuf
export_config {
}
```

- batch_size: 导出模型的batch_size，默认是-1，即可以接收任意batch_size
- exporter_type: 导出类型, best | final | latest | none，默认final
  - best 导出最好的模型
  - final 训练结束后导出
  - latest 导出最新的模型
  - none 不导出
- best_exporter_metric: 当exporter_type为best的时候，确定最优导出模型的metric，注意该metric要在eval_config的metrics_set设置了才行
- metric_bigger: 确定最优导出模型的metric是越大越好，还是越小越好，默认是越大越好
- exports_to_keep: 当exporter_type为best或lastest时，保留n个最好或最新的模型，默认为1
  ```protobuf
  export_config {
    exporter_type: "best"
    best_exporter_metric: "auc"
    exports_to_keep: 1
  }
  ```
- multi_placeholder: 使用一个placeholder还是多个placeholder。默认为true，即对每个特征使用单个placeholder
- multi_value_fields: 针对tagFeature，指定一个字段集合，使得导出的placeholder可以接收二维数组，而不是训练时用的字符串类型，这样可以节省字符串拆分和类型转换的时间。
  ```protobuf
  export_config {
    multi_value_fields {
       input_name: ["field1", "field2", "field3"]
    }
  }
  ```
- placeholder_named_by_input: true时利用data_config.input_fields.input_name来命令每个placeholder，false时每个placeholder名字为"input_X"，"X"为data_config.input_fields的编号(0-input_num)。默认为False
- asset_files: 需要导出的asset文件, 可设置多个
- enable_early_stop: 根据early_stop_func的返回值判断是否要提前结束训练
  - 示例:
    - samples/model_config/custom_early_stop_on_taobao.config
    - samples/model_config/multi_tower_early_stop_on_taobao.config
  - 应用场景:
    - 数据量比较小时，需要训练多个epoch时, 打开early_stop可以防止过拟合;
    - 使用[超参搜索](./automl/pai_nni_hpo.md)时, 打开可以提前终止收敛比较差的参数, 显著提升搜索效率
- early_stop_func: 判断是否要提前结束训练的函数
  - 返回值:
    - True, 结束训练
    - False, 继续训练
  - 默认值:
    - metric_bigger为true时, easy_rec.python.compat.early_stopping.stop_if_no_increase_hook
    - metric_bigger为false时, easy_rec.python.compat.early_stopping.stop_if_no_decrease_hook
  - 自定义early_stop_func:
    - 示例: easy_rec.python.test.custom_early_stop_func.custom_early_stop_func
    - 参数: 框架传入两个参数
      - eval_results: 模型评估结果
      - func_param: 自定义参数(即export_config.early_stop_params)
- max_check_steps: 训练max_check_steps之后评估指标没有改善，即停止训练; 仅适用于内置early_stop_func, 不适用于自定义early_stop_func
  - stop_if_no_increase_hook: 对应max_steps_without_increase, 当间隔max_check_steps训练步数评估指标没有提升，即停止训练
  - stop_if_no_decrease_hook: 对应max_steps_without_decrease, 当间隔max_check_steps训练步数评估指标没有下降, 即停止训练
- early_stop_params: 传递给early_stop_func的自定义参数

### 导出命令

#### Local

```bash
python -m easy_rec.python.export --pipeline_config_path dwd_avazu_ctr_deepmodel.config --export_dir ./export --export_done_file EXPORT_DONE
```

- --pipeline_config_path: config文件路径
- --model_dir: 如果指定了model_dir将会覆盖config里面的model_dir，一般在周期性调度的时候使用
- --export_dir: 导出的目录
- --export_done_file: 导出完成标志文件名, 导出完成后，在导出目录下创建一个文件表示导出完成了
- --clear_export: 删除旧的导出文件目录

#### PAI

```sql
pai -name easy_rec_ext -project algo_public
-Dconfig=oss://easyrec/easy_rec_test/dwd_avazu_ctr_deepmodel_ext.config
-Dcmd=export
-Dexport_dir=oss://easyrec/easy_rec_test/export
-Dcluster='{"worker" : {"count":1, "cpu":1000, "memory":40000}}'
-Darn=acs:ram::xxx:role/ev-ext-test-oss
-Dbuckets=oss://easyrec/
-DossHost=oss-cn-beijing-internal.aliyuncs.com
```

- -Dconfig: 同训练
- -Dcmd: export 模型导出
- -Dexport_dir: 导出的目录
- -Dcheckpoint_path: 可选参数，使用指定的checkpoint_path导出
- -Darn: rolearn 注意这个的arn要替换成客户自己的。可以从dataworks的设置中查看arn。
- -DossHost: ossHost地址
- -Dbuckets: config所在的bucket和保存模型的bucket; 如果有多个bucket，逗号分割
- 如果是pai内部版,则不需要指定arn和ossHost, arn和ossHost放在-Dbuckets里面
  - -Dbuckets=oss://easyrec/?role_arn=acs:ram::xxx:role/ev-ext-test-oss&host=oss-cn-beijing-internal.aliyuncs.com
- -Dextra_params: 其它参数, 没有在pai -name easy_rec_ext中定义的参数, 可以通过extra_params传入, 如:
  - --export_done_file: 导出完成标志文件名, 导出完成后，在导出目录下创建一个文件表示导出完成了
  - --clear_export: 删除旧的导出文件目录
  - --place_embedding_on_cpu: 将embedding相关的操作放在cpu上，有助于提升模型在gpu环境下的推理速度
  - --asset_files: 需要导出的asset文件路径, 可设置多个, 逗号分隔；
    - 如果需要导出到assets目录的子目录下，使用`${target_path}:${source_path}`的格式；（从版本0.8.7开始支持）
    - e.g. '--asset_files custom_fg_lib/fg.json:oss://${bucket}/path/to/fg.json'
- 模型导出之后可以使用(EasyRecProcessor)\[./predict/在线预测.md\]部署到PAI-EAS平台

### 双塔召回模型

如果是双塔召回模型(如dssm, mind等), 模型导出之后, 一般还需要进行模型切分和索引构建, 才能使用(EasyRecProcessor)\[./predict/在线预测.md\]部署到PAI-EAS上.

#### 模型切分

```sql
pai -name easy_rec_ext
-Dcmd='custom'
-DentryFile='easy_rec/python/tools/split_model_pai.py'
-Dversion='{easyrec_version}'
-Dbuckets='oss://{oss_bucket}/'
-Darn='{oss_arn}'
-DossHost='oss-{region}-internal.aliyuncs.com'
-Dcluster='{{
    \\"worker\\": {{
        \\"count\\": 1,
        \\"cpu\\": 100
    }}
}}'
-Dextra_params='--model_dir=oss://{oss_bucket}/dssm/export/final --user_model_dir=oss://{oss_bucket}/dssm/export/user --item_model_dir=oss://{oss_bucket}/dssm/export/item --user_fg_json_path=oss://{oss_bucket}/dssm/user_fg.json --item_fg_json_path=oss://{oss_bucket}/dssm/item_fg.json';
```

- -Dextra_params:
  - --model_dir: 待切分的saved_model目录
  - --user_model_dir: 切分好的用户塔模型目录
  - --item_model_dir: 切分好的物品塔模型目录
  - --user_fg_json_path: 用户塔的fg json
  - --item_fg_json_path: 物品塔的fg json

#### 物品Embedding预测和索引构建

```sql
pai -name easy_rec_ext
-Dcmd='predict'
-Dsaved_model_dir='oss://{oss_bucket}/dssm/export/item/'
-Dinput_table='odps://{project}/tables/item_feature_t'
-Doutput_table='odps://{project}/tables/dssm_item_embedding'
-Dreserved_cols='item_id'
-Doutput_cols='item_emb string'
-Dmodel_outputs='item_emb'
-Dbuckets='oss://{oss_bucket}/'
-Darn='{oss_arn}'
-DossHost='oss-{region}-internal.aliyuncs.com'
-Dcluster='{{
    \\"worker\\": {{
        \\"count\\": 16,
        \\"cpu\\": 600,
        \\"memory\\": 10000
    }}
}}';
```

```sql
pai -name easy_rec_py3_ext
-Dcmd='custom'
-DentryFile='easy_rec/python/tools/faiss_index_pai.py'
-Dtables='odps://{project}/tables/dssm_item_embedding'
-Dbuckets='oss://{oss_bucket}/'
-Darn='{oss_arn}'
-DossHost='oss-{region}-internal.aliyuncs.com'
-Dcluster='{{
    \\"worker\\": {{
        \\"count\\": 1,
        \\"cpu\\": 100
    }}
}}'
-Dextra_params='--index_output_dir=oss://{oss_bucket}/dssm/export/user';
```

- -Dtables: 物品向量表
- -Dextra_params:
  - --index_output_dir: 索引输出目录, 一般设置为已切分好的用户塔模型目录，便于用EasyRec Processor部署
  - --index_type: 索引类型，可选 IVFFlat | HNSWFlat，默认为 IVFFlat
  - --ivf_nlist: 索引类型为IVFFlat是，聚簇的数目
  - --hnsw_M: 索引类型为HNSWFlat的索引参数M
  - --hnsw_efConstruction: 索引类型为HNSWFlat的索引参数efConstruction
