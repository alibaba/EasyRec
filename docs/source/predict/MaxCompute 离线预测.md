# MaxCompute 离线预测

### 前置条件：

- 模型训练
- 模型导出

### 离线预测

```bash
drop table if exists ctr_test_output;
pai -name easy_rec_ext
-Dcmd=predict
-Dcluster='{"worker" : {"count":5, "cpu":1000,  "memory":40000, "gpu":0}}'
-Darn=acs:ram::xxx:role/aliyunodpspaidefaultrole
-Dbuckets=oss://easyrec/
-Dsaved_model_dir=oss://easyrec/easy_rec_test/experiment/export/1597299619
-Dinput_table=odps://pai_online_project/tables/ctr_test_input
-Doutput_table=odps://pai_online_project/tables/ctr_test_output
-Dexcluded_cols=label
-Dreserved_cols=ALL_COLUMNS
-Dbatch_size=1024
-DossHost=oss-cn-beijing-internal.aliyuncs.com;
```

- cluster: 这里cpu:1000表示是10个cpu核；核与内存的关系设置1:4000，一般不超过40000；gpu设置为0，表示不用GPU推理。
- saved_model_dir: 导出的模型目录
- output_table: 输出表，不需要提前创建，会自动创建
- excluded_cols: 预测模型不需要的columns，比如labels
- selected_cols: 预测模型需要的columns，selected_cols和excluded_cols不能同时使用
- reserved_cols: 需要copy到output_table的columns, 和excluded_cols/selected_cols不冲突，如果指定ALL_COLUMNS，则所有的column都被copy到output_table
- batch_size: minibatch的大小
- arn: rolearn  注意这个的arn要替换成客户自己的。可以从dataworks的设置中查看arn。
- ossHost: ossHost地址
- buckets: config所在的bucket和保存模型的bucket; 如果有多个bucket，逗号分割
  - 如果是pai内部版,则不需要指定arn和ossHost, arn和ossHost放在-Dbuckets里面
  ```
    -Dbuckets=oss://easyrec/?role_arn=acs:ram::xxx:role/aliyunodpspaidefaultrole&host=oss-cn-beijing-internal.aliyuncs.com
  ```
- output_cols: 指定输出表里面的column name和type:
  - 默认是"probs double"
  - 如果有多列，用逗号分割, 如:
    ```sql
      -Doutput_cols='probs double,embedding string'
    ```
  - 默认column name和saved_model导出字段名称一致
    - 如果不一致, 请使用model_outputs指定对应的导出字段名称
  - 模型导出的字段类型和MaxCompute表字段类型对应关系:
    - float/double : double
    - string : string
    - int32/int64 : bigint
    - array : string(json format)
    - 其他类型: 暂不支持
  - 二分类模型(要求num_class=1)，导出字段:logits、probs，对应: sigmoid之前的值/概率
  - 回归模型，导出字段: y，对应: 预测值
  - 多分类模型(num_class > 1)，导出字段:
    - logits: string(json), softmax之前的vector, shape\[num_class\]
    - probs: string(json), softmax之后的vector, shape\[num_class\]
      - 如果一个分类目标是is_click, 输出概率的变量名称是probs_is_click
      - 多目标模型中有一个回归目标是paytime，那么输出回归预测分的变量名称是：y_paytime
    - logits_y: logits\[y\], float, 类别y对应的softmax之前的概率
    - probs_y: probs\[y\], float, 类别y对应的概率
    - y: 类别id, = argmax(probs_y), int, 概率最大的类别
    - 示例:
    ```sql
      -Doutput_cols='logits string,probs string,logits_y double,probs_y double,y bigint'
    ```
  - 查看导出字段:
    - 如果不确定saved_model的导出字段的信息, 可以通过下面的命令查看:
    ```bash
       saved_model_cli show --all --dir ./data/test/inference/fm_export/
    ```
    - 输出信息:
    ```text
       MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

       signature_def['serving_default']:
         The given SavedModel SignatureDef contains the following input(s):
           inputs['adgroup_id'] tensor_info:
               dtype: DT_STRING
               shape: (-1)
               name: input_3:0

       ...

       The given SavedModel SignatureDef contains the following output(s):
         outputs['logits'] tensor_info:
             dtype: DT_FLOAT
             shape: (-1)
             name: Squeeze:0
         outputs['probs'] tensor_info:
             dtype: DT_FLOAT
             shape: (-1)
             name: Sigmoid:0
       Method name is: tensorflow/serving/predict
    ```
    - 可以看到导出的字段包括:
      - logits, float
      - probs, float
- model_outputs: 导出saved_model时模型的导出字段，可以不指定，默认和output_cols一致
  - 如果output_cols和model_outputs不一致时需要指定，如:
    ```sql
      -Doutput_cols='score double' -Dmodel_outputs='probs'
    ```
  - 多列示例:
    ```sql
    -Doutput_cols='scores double,user_embed string'
    -Dmodel_outputs='probs,embedding'
    ```
    - 格式: ","分割
    - 顺序: column_name和导出字段名一一对应
- lifecycle: output_table的lifecycle

### 输出表schema:

包含reserved_cols和output_cols
