# RTP FG

- RTP FG: RealTime Predict Feature Generation, 解决实时预测需要的特征工程需求. 特征工程在推荐链路里面也占用了比较长的时间.

- RTP FG能够以比较高的效率生成一些复杂的交叉特征，如match feature和lookup feature.离线训练和在线预测的时候通过使用同一套c++代码保证离线在线的一致性.

- 其生成的特征可以接入EasyRec进行训练，从RTP FG的配置(fg.json)可以生成EasyRec的配置文件(pipeline.config).

- 线上部署的时候提供带FG功能的[EasyRec Processor](../predict/processor.md)一键部署.

### 训练

#### 编写配置 [fg.json](https://easyrec.oss-cn-beijing.aliyuncs.com/rtp_fg/fg.json)

- 包含了features配置和全局配置两个部分,  示例:

```json
{
  "features": [
     {"expression": "user:user_id", "feature_name": "user_id", "feature_type":"id_feature", "value_type":"String", "combiner":"mean", "hash_bucket_size": 100000, "embedding_dim": 16, "group":"user"},
     {"expression": "user:cms_segid", "feature_name": "cms_segid", "feature_type":"id_feature", "value_type":"String", "combiner":"mean", "hash_bucket_size": 100, "embedding_dim": 16, "group":"user"},
     ...
     {"expression": "item:price", "feature_name": "price", "feature_type":"raw_feature", "value_type":"Integer", "combiner":"mean", "group":"item"},
     {"expression": "item:pid", "feature_name": "pid", "feature_type":"id_feature", "value_type":"String", "combiner":"mean", "hash_bucket_size": 100000, "embedding_dim": 16, "group":"item"},
     {"expression": "user:tag_category_list", "feature_name": "user_tag_cate", "feature_type":"id_feature", "hash_bucket_size":100000, "group":"user"},
     {"map": "user:tag_brand_list", "key":"item:brand", "feature_name": "combo_brand", "feature_type":"lookup_feature",  "needDiscrete":true, "hash_bucket_size":100000, "group":"combo"},
     {"map": "user:tag_category_list", "key":"item:cate_id", "feature_name": "combo_cate_id", "feature_type":"lookup_feature",  "needDiscrete":true, "hash_bucket_size":10000, "group":"combo"}
 ],


 "reserves": [
   "user_id", "campaign_id", "clk"
 ]
}
```

- Feature配置说明：

  - [id_feature](./fg_docs/IdFeature.md)

    - is_multi: id_feature是否是多值属性

      - 默认是true, 转换成EasyRec的config时会转成TagFeature

      - 如果设成false, 转换成EasyRec的config时会转成IdFeature, 可以减少字符串分割的开销

      - 多值分隔符使用chr(29)\[ctrl+v ctrl+\], 即"\\u001D".

      - [多值类型说明](./fg_docs/mutiValues.md)

    - vocab_file: 词典文件路径，根据词典将对应的输入映射成ID.

    - vocab_list: 词典list，根据词典将对应的输入映射成ID.

    - num_buckets: 当输入是unsigned int类型的时候，并且输入有界的时候，可以指定num_bucket为输入的最大值.

    - hash_bucket_size: 对应EasyRec feature_config.features的hash_bucket_size.

      - 和vocab_file, vocab_list相比，优势是不需要词典，词典可以是不固定的.

      - 劣势是需要设置的容量比较大，容易导致hash冲突.

    - embedding_dimension/embedding_dim: 对应EasyRec feature_config.features里面的embedding_dim.

  - [raw_feature](./fg_docs/RawFeature.md)

    - bucketize_boundaries: 会生成离散化的结果, 在生成EasyRec config的时候:

    - 设置feature_config.features.num_buckets = len(boundaries) + 1

    - value_dimension > 1时, feature_type = TagFeature

    - value_dimension = 1时, feature_type = IdFeature

    - boundaries: 生成的还是连续值，但在生成EasyRec config的时候:

    ```
    会配置离散化的bucket, 如:
    feature_config: {
      features: {
        input_names: "hour"
        feature_type: RawFeature
        boundaries: [1,5,9,15,19,23]
        embedding_dim: 16
      }
    }
    ```

    - 设置bucketize_boundaries/boundaries的同时需要设置embedding_dimension.

    - value_dimension: 连续值的维度，>1时表示有多个连续值, 也就是一个向量.

      - 比如ctr_1d,ctr_2d,ctr_3d,ctr_12d可以放在一个RawFeature里面.
      - 该选项对生成数据有影响.
      - 该选项对生成EasyRec config也有影响, 对应到[feature_config.raw_input_dim](../proto.html#protos.FeatureConfig)

  - [combo_feature](./fg_docs/ComboFeature.md)

    - 需要设置embedding_dimension和hash_bucket_size.
      方法一：在fg中生成combo特征，见[combo_feature](./fg_docs/ComboFeature.md)

    ```
    {"expression" : ["user:user_id", "user:occupation"], "feature_name" : "combo__occupation_age_level", "feature_type" : "combo_feature", "hash_bucket_size": 10, "embedding_dim": 16}

    ```

    - fg.json需进行三项配置，生成三列数据

    方法二：在参与combo的特征配置中加入extra_combo_info配置，fg会生成两列数据，在easyrec层面进行combo.

    ```
     {"expression": "user:user_id", "feature_name": "user_id", "feature_type":"id_feature", "value_type":"String", "combiner":"mean", "hash_bucket_size": 100000, "embedding_dim": 16, "group":"user"},
     {"expression": "user:occupation", "feature_name": "occupation", "feature_type":"id_feature", "value_type":"String", "combiner":"mean", "hash_bucket_size": 10, "embedding_dim": 16, "group":"user",
       "extra_combo_info": {
         "final_feature_name": "combo__occupation_age_level",
         "feature_names": ["user_id"],
         "combiner":"mean", "hash_bucket_size": 10, "embedding_dim": 16
       }
     }
    ```

    - 最终会生成两列数据（user_id和occupation），config中生成三个特征的配置，分别是user_id，occupation，combo\_\_occupation_age_level.
    - final_feature_name: 该combo特征的名字.
    - feature_names: 除当前特征外，参与combo的特征，至少一项.
    - combiner, hash_bucket_size, embedding_dim 配置与上述一致.

  - [lookup_feature](./fg_docs/LookupFeature.md)

    - 单层查找, 根据id(如item_id, item_category_id等)查找对应的value.

  - [match_feature](./fg_docs/MatchFeature.md)

    - 双层查找, 根据category和item_id查找value.

    - match feature里面多值分隔符可以使用chr(29) (ctrl+v ctrl+\])或者逗号\[,\]， 如:

    ```
      50011740^107287172:0.2^]36806676:0.3^]122572685:0.5|50006842^16788816:0.1^]10122:0.2^]29889:0.3^]30068:19
    ```

    - needWeighting: 生成特征权重，即kv格式, kv之间用\[ctrl+v ctrl+e\]分割, 转换成TagFeature.

  - [sequence_feature](./fg_docs/SequenceFeature.md)

    - 序列特征用于对用户行为建模, 通常应用于DIN和Transformer模型当中

    - sequence_pk: 行为序列的特征名, 如点击序列, 购买序列等, 一般保存在item侧, 如user:clk_seq_50

      - 离线格式: item_id和属性拼在一起, 通过#分隔

        - 示例: item\_\_id:11#item\_\_price:2.0;item_id:22#item\_\_price:4.0

      - 在线格式: 只保留item_id

        - 示例: 11;22

    - sequence_table: 一般都是item, online serving时从item表中根据item_id查询item信息, 离线时没有用

    - Note: item_seq(如item的图片列表)目前还不支持

  - [overlap_feature](./fg_docs/OverLapFeature.md)

  - 针对EasyRec的扩展字段:

    - group: feature所属的分组

      - 对于WideAndDeep/DeepFM是wide/deep.

      - 对于MultiTower可以自定义分组名称，如user/item/combo.

    - combiner: 默认是mean, 也可以是sum.

      - 影响数据生成和 EasyRec feature_config 生成, 主要是多值Feature.

- 全局配置说明:

  - reserves: 要在最终表里面要保存的字段，通常包括label, user_id, item_id等

  - separator: sparse格式里面，特征之间的分隔符，不指定默认是","，

    - 训练时，对稠密格式没有影响，对稀疏格式有影响
    - 预测时，item feature在redis里面存储的是稀疏格式，因此是有影响的

    ```
    i_item_id:10539078362,i_seller_id:21776327,...
    ```

  - kv_separator: 多值有权重特征的分隔符，如”体育:0.3|娱乐:0.2|军事:0.5”，不指定默认None，即没有权重

  - model_dir: 模型目录，仅仅影响EasyRec config生成.

  - num_steps: 训练的轮数，仅仅影响EasyRec config生成.

  - embedding_dim: 全局的embedding dimension.

    - 适合DeepFM等需要所有的feature都使用统一的embedding_dim.

    - 如果feature字段没有单独设置embedding_dimension, 将使用统一的embedding_dim.

    - 配置里面的embedding_dim会覆盖从命令行easy_rec.python.tools.convert_rtp_fg传入的embedding_dim.

  - model_type: 模型的类型，当前支持WideAndDeep/MultiTower/DeepFM.

    - 暂未支持的EasyRec模型，可以不指定model_type，在生成EasyRec config之后添加相应的部分.

  - label_fields: label数组，针对多目标模型需要设置多个label fields.

  - model_path: 定义模型部分的config文件, 适用于暂未支持的EasyRec模型或自定义模型.

  - edit_config_json: 对EasyRec config的修改, 如修改dnn的hidden_units

  ```
  "edit_config_json": [{"model_config.wide_and_deep.dnn.hidden_units": [48, 24]}]
  ```

#### 上传数据(如果已经有数据，可以跳过这一步)

支持两种格式: 稀疏格式和稠密格式, 根据表的schema自动识别是哪一种格式, 包含user_feature和item_feature则识别成稀疏格式.

- 稀疏格式的数据: user特征, item特征, context特征各放一列；特征在列内以kv形式存储, 如：

| label | user_id | item_id | context_feature | user_feature                                                    | item_feature                                       |
| ----- | ------- | ------- | --------------- | --------------------------------------------------------------- | -------------------------------------------------- |
| 0     | 122017  | 389957  |                 | tag_category_list:4589,new_user_class_level:,...,user_id:122017 | adgroup_id:539227,pid:430548_1007,...,cate_id:4281 |

提供了在任何项目下都可以访问两张样例表

```
pai_online_project.taobao_train_input
pai_online_project.taobao_test_input
```

- 稠密格式的数据，每个特征是单独的一列，如：

| label | user_id | item_id | tag_category_list | new_user_class_level | age_level |
| ----- | ------- | ------- | ----------------- | -------------------- | --------- |
| 1     | 122017  | 389957  | 4589              |                      | 0         |

```sql
  drop table if exists taobao_train_input_dense;
  create table taobao_train_input_dense(label bigint, user_id string, item_id string, tag_category_list bigint, ...);
```

- **Note:** 特征列名可以加上prefix: **"user\_\_", "item\_\_", "context\_\_"**

```
  如: 列名ctx_position也可以写成 context__ctx_position
```

#### 生成样本

- 下载rtp_fg [jar ](https://easyrec.oss-cn-beijing.aliyuncs.com/deploy/fg_on_odps-1.3.59-jar-with-dependencies.jar)包
- 生成特征

```sql
add jar target/fg_on_odps-1.3.59-jar-with-dependencies.jar -f;
add file fg.json -f;

set odps.sql.planner.mode=sql;
set odps.isolation.session.enable=true;
set odps.sql.counters.dynamic.limit=true;

drop table if exists taobao_fg_train_out;
create table taobao_fg_train_out(label bigint, user_id string, item_id string,  features string);
-- dataworks内运行，注意需要带有resource_reference这一行
--@resource_reference{"fg_on_odps-1.3.59-jar-with-dependencies.jar"}
jar -resources fg_on_odps-1.3.59-jar-with-dependencies.jar,fg.json -classpath fg_on_odps-1.3.59-jar-with-dependencies.jar com.taobao.fg_on_odps.EasyRecFGMapper -i pai_online_project.taobao_train_input -o taobao_fg_train_out -f fg.json;
drop table if exists taobao_fg_test_out;
create table taobao_fg_test_out(label bigint, user_id string, item_id string,  features string);
-- dataworks内运行，注意需要带有resource_reference这一行
--@resource_reference{"fg_on_odps-1.3.59-jar-with-dependencies.jar"}
jar -resources fg_on_odps-1.3.59-jar-with-dependencies.jar,fg.json -classpath fg_on_odps-1.3.59-jar-with-dependencies.jar com.taobao.fg_on_odps.EasyRecFGMapper -i pai_online_project.taobao_test_input -o taobao_fg_test_out -f fg.json;

--下载查看数据(可选)
tunnel download taobao_fg_test_out taobao_fg_test_out.txt -fd=';';
```

- EasyRecFGMapper参数格式:
  - -i, 输入表
    - 支持分区表，分区表可以指定partition，也可以不指定partition，不指定partition时使用所有partition
    - **分区格式示例:** my_table/day=20201010,sex=male
    - 可以用多个-i指定**多个表的多个分区**
    - 支持添加project，示例：project.table/ds=xxx
  - -o, 输出表，如果是分区表，一定要指定分区，只能指定一个输出表
  - -f, fg.json
  - -m, mapper memory的大小，默认可以不设置
- EasyRecFGMapper会自动判断是**稠密格式**还是**稀疏格式**
  - 如果表里面有user_feature和item_feature字段，那么判定是稀疏格式
  - 否则，判定是稠密格式
- 生成的特征示例(taobao_fg_train_out):

| label | user_id | item_id | features                                                                                                      |
| ----- | ------- | ------- | ------------------------------------------------------------------------------------------------------------- |
| 0     | 336811  | 100002  | user_id_100002^Bcms_segid_5^Bcms_group_id_2^Bage_level_2^Bpvalue_level_1^Bshopping_level_3^Boccupation_1^B... |

#### 从配置文件\[fg.json\]生成EasyRec的config

从Git克隆EasyRec

```bash
git clone https://github.com/alibaba/EasyRec.git
```

```python
python -m easy_rec.python.tools.convert_rtp_fg  --label clk --rtp_fg fg.json --model_type multi_tower --embedding_dim 10  --output_path fg.config --selected_cols "label,features"
```

多目标模型写法

```
python -m easy_rec.python.tools.convert_rtp_fg  --label is_product_detail is_purchase --rtp_fg fg.json --model_type dbmtl --embedding_dim 10  --output_path fg.config --selected_cols "is_product_detail,is_purchase,features"
```

- --model_type: 模型类型, 可选: multi_tower, deepfm, essm, dbmtl 其它模型暂时不能设置，需要在生成的config里面增加model_config的部分

- --embedding_dim: embedding dimension, 如果fg.json里面的feature没有指定embedding_dimension, 那么将使用该选项指定的值

- --batch_size: batch_size, 训练时使用的batch_size

- --label: label字段, 可以指定多个

- --num_steps: 训练的步数,默认1000

- --output_path: 输出的EasyRec config路径

- --separator: feature之间的分隔符, 默认是CTRL_B(u0002)

- --selected_cols: 指定输入列，包括label、\[sample_weight\]和features，其中label可以指定多列，表示要使用多个label(一般是多任务模型),  最后一列必须是features, 如:

  ```
  label0,label1,sample_weight,features
  ```

  - 注意不要有**空格**，其中 sample_weight 列是可选的，可以没有

- --incol_separator: feature内部的分隔符，即多值分隔符，默认是CTRL_C(u0003)

- --input_type: 输入类型

  - OdpsRTPInput表示在MaxCompute上使用;
  - RTPInput, 本地使用, 使用RTPInput时需要指定训练时用到的对应的列的id, 如:
    ```
    --selected_cols=0,4
    ```
    - 其中第0列是label, 第4列是features
    - 还需要指定--rtp_separator，表示label和features之间的分隔符, 默认是";"
  - HiveRTPInput, 用于在DataScience上使用

- --train_input_path, 训练数据路径

  - MaxCompute上不用指定，在训练的时候指定

- --eval_input_path, 评估数据路径

  - MaxCompute上不用指定，在训练的时候指定

#### 启动训练

- 上传fg.config到oss
- 启动训练

```sql
pai -name easy_rec_ext
-Dversion='0.4.5'
-Dcmd=train
-Dconfig=oss://bucket-name/easy_rec_test/fg.config
-Dtrain_tables=odps://project-name/tables/taobao_fg_train_out
-Deval_tables=odps://project-name/tables/taobao_fg_test_out
-Dcluster='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":3, "cpu":1000, "gpu":100, "memory":40000}}'
-Darn=acs:ram::xxx:role/ev-ext-test-oss
-Dbuckets=oss://bucket-name/
-DossHost=oss-cn-xxx.aliyuncs.com
-Deval_method=separate;
```

- 参数说明: [请参考](../train.md#on-pai)

#### 模型导出

```sql
pai -name easy_rec_ext
    -Dversion='0.4.5'
    -Dcmd=export
    -Dconfig=oss://easyrec/easy_rec_test/fg.config
    -Dexport_dir=oss://<bucket-name>/export_dir
    -Dbuckets=oss://<bucket-name>/
    -Darn=acs:ram::xxx:role/aliyunodpspaidefaultrole
    -DossHost=oss-hangzhou-internal.aliyuncs.com
    -Dedit_config_json='{"export_config.multi_placeholder":true, "feature_config.features[:].max_partitions":1}';

```

- 参数说明: [请参考](../export.md#pai)
- 注意事项:
  - 请检查fg.config, 保证导出的模型是支持多个placeholder的输入\[每个特征一个placeholder\]

    ```
    export_config {
      multi_placeholder: true
    }
    ```

    如果不是, 可以通过-Dedit_config_json='{"export_config.multi_placeholder":true}' 进行修改

  - 如果有设置feature_config.features.max_partitions, 请加入下面的命令重置:

    - -Dedit_config_json='{"feature_config.features\[:\].max_partitions":1}'进行修改, 可以获得更好的性能

#### 增加特征

- 增加特征可以用原来的样本表A left outer join 新增的特征表B 生成表C

```
  create table C
  as select * from A
  left outer join B
  on A.req_id = B.req_id and A.item_id = B.item_id
```

- 表C使用增量incre_fg.json生成表incre_fea_table, incre_fg.json定义了新增的特征

```
  jar -resources fg_on_odps-1.3.59-jar-with-dependencies.jar,incre_fg.json -classpath fg_on_odps-1.3.59-jar-with-dependencies.jar com.taobao.fg_on_odps.EasyRecFGMapper -i taobao_test_input -o taobao_fg_test_out -f incre_fg.json;
```

- 生成新的样本表D:

```
  create new_feature_table as
  select A.*, wm_concat(fea_table.features, chr(2), incre_fea_table.features) as features
  from A
    inner join fea_table
  on A.req_id = fea_table.req_id and A.item_id = fea_table.item_id
    inner join incre_fea_table
  on A.req_id = incre_fea_table.req_id and A.req_id = incre_fea_table.item_id
```

#### 特征筛选

- 可以筛选fg.json里面的部分特征用于训练

- 方法: 在fg.config的model_config.feature_groups里面把不需要的特征注释掉即可

### 预测

#### 服务部署

- 部署的 service.json 示例如下

```shell
bizdate=$1
cat << EOF > echo.json
{
  "name":"ali_rec_rnk",
  "metadata": {
    "resource": "eas-r-xxxx",
    "cpu": 8,
    "memory": 20000,
    "instance": 2,
    "rpc": {
      "enable_jemalloc": 1,
      "max_queue_size": 100
    }
  },
  "model_config": {
    "remote_type": "hologres",
    "url": "postgresql://<AccessKeyID>:<AccessKeySecret>@<域名>:<port>/<database>",
    "tables": [{"name":"<schema>.<table_name>","key":"<index_column_name>","value": "<column_name>"}],
    "period": 2880,
    "fg_mode": "tf",
    "outputs":"probs_ctr,probs_cvr",
  },
  "processor": "easyrec-1.5",
  "storage": [
    {
      "mount_path": "/home/admin/docker_ml/workspace/model/",
      "oss": {
        "endpoint": "oss-cn-hangzhou-internal.aliyuncs.com",
        "path": "oss://easyrec/ali_rec_sln_acc_rnk/20221122/export/final_with_fg"
      }
    }
  ]
}

EOF
# 执行部署命令。
eascmd -i <AccessKeyID>  -k  <AccessKeySecret>   -e <EndPoint> create echo.json
eascmd -i <AccessKeyID>  -k  <AccessKeySecret>   -e <EndPoint> update ali_rec_rnk -s echo.json

```

- processor: easyrec processor, 目前最新的版本为easyrec-1.5, [历史版本](../predict/processor.md#release).

- model_config: eas 部署配置。主要控制把 item 特征加载到内存中。目前数据源支持redis和holo

  - period: item feature reload period, 单位minutes
  - url: holo url
  - fg_mode: 支持tf和normal两种模式, tf模式表示fg是以TF算子的方式执行的, 性能更好
  - tables: item特征存储在hologres表里面, 支持分多个表存储
    - key: 必填, itemId列的名字;
    - value: 可选，需要加载的列名, 多个列名之间用逗号(,)分割;
    - condition: 可选，where子语句支持筛选item, 如itemId \< 10000;
    - timekey: 可选，用于item的增量更新，支持的格式: timestamp和int
    - static: 可选, 表示是静态特征，不用周期性更新
    - 支持多个item表, 如果多张表有重复的列, 后面的表覆盖前面的表
    - hologres表里面每一列存储一个item特征,示例:
      <table class="docutils" border=1>
       <tr><th>adgroup_id</th><th>cate_id</th><th>campaign_id</th><th>customer</th><th>brand</th><th>price</th></tr>
       <tr><td>100038</td><td>5480</td><td>37448</td><td>117182</td><td>6077</td><td>4</td></tr>
       <tr><td>100039</td><td>10344</td><td>122588</td><td>96590</td><td>14287</td><td>97</td></tr>
       <tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>
      </table>

- storage: 将oss的模型目录mount到docker的指定目录下

  - mount_path: docker内部的挂载路径, 与示例保持一致即可
  - 配置了storage就不需要配置model_path了
  - 优点: 部署速度快

- model_path: 将模型拷贝到docker内部

  - 缺点: 部署速度慢, 需要将模型保存到docker内部
  - 建议仅在无法通过storage挂载的情况下使用model_path

- 其它参数是所有EAS服务通用的, 请参考[EAS文档](https://help.aliyun.com/zh/pai/user-guide/parameters-of-model-services).

#### 客户端请求

和TFRequest类似, EasyRec Processor也是使用ProtoBuffer 作为传输协议. [proto文件定义](https://github.com/pai-eas/eas-java-sdk/blob/master/src/main/proto/easyrec_predict.proto). Java客户端可以通过PAI-EAS Java SDK调用服务, 在pom.xml里面加入:

```
<dependency>
  <groupId>com.aliyun.openservices.eas</groupId>
  <artifactId>eas-sdk</artifactId>
  <version>2.0.9</version>
</dependency>
```

代码参考：

```java
import com.aliyun.openservices.eas.predict.http.*;
import com.aliyun.openservices.eas.predict.request.EasyRecRequest;

PredictClient client = new PredictClient(new HttpConfig());

// 使用网络直连功能, 为了提升吞吐降低RT, 建议使用网络直连
// Endpoint需要根据相应的region进行修改
client.setDirectEndpoint("pai-eas-vpc.cn-hangzhou.aliyuncs.com");

// 通过普通网关访问时，需要使用以用户UID开头的Endpoint
//    在PAI-EAS控制台服务的调用信息中可以获得该信息
client.setEndpoint("xxxxxxx.vpc.cn-hangzhou.pai-eas.aliyuncs.com");

client.setModelName("ali_rec_rnk");
// 设置服务Token信息
client.setToken("atxjzk****");

EasyRecRequest easyrecRequest = new EasyRecRequest(separator);
easyrecRequest.appendUserFeatureString(userFeatures);
easyrecRequest.appendContextFeatureString(contextFeatures);
easyrecRequest.appendItemStr(itemIdStr, ",");

PredictProtos.PBResponse response = client.predict(easyrecRequest);

for (Map.Entry<String, PredictProtos.Results> entry : response.getResultsMap().entrySet()) {
    String key = entry.getKey();
    PredictProtos.Results value = entry.getValue();
    System.out.print("key: " + key);
    for (int i = 0; i < value.getScoresCount(); i++) {
        System.out.format("value: %.6g\n", value.getScores(i));
    }
}
```

- client.setDirectEndpoint: [网络直连](https://help.aliyun.com/zh/pai/user-guide/call-a-service-over-the-vpc-direct-connection-channel)可以减少网络传输时间, 显著提升性能

  - 请从上述文档查看不同region对应的direct endpoint地址

- EasyRecRequest类方法列表

  <table class="docutils" border=1>
    <tr><th>方法</th><th>描述</th></tr>
    <tr><td>setSeparator(String sep)</td><td>设置user特征分隔符, 默认是"\u0002", 即CTRL_B</td></tr>
    <tr><td>addUserFeature(String key, T value)</td><td>增加一个user特征, key是特征名, value是特征值, value支持的类型包括String, float, long, int</td></tr>
    <tr><td>appendUserFeatureString</td><td>增加一组user特征,特征之间以separator分隔,特征内部使用":"分隔特征名和特征值</td></tr>
    <tr><td>appendItemId(String itemId)</td><td>增加一个itemId</td></tr>
    <tr><td>appendItemStr(String itemIdStr)</td><td>增加一组itemIds, 以","分隔</td></tr>
    <tr><td>addContextFeature(String key, List<Object> contextFeatures)</td><td>增加一个context特征, key是特征名, contextFeatures是特征值的列表, 列表中的元素和itemId一一对应</td></tr>
    <tr><td>addContextFeatureString(String contextStr)</td><td>增加一个context特征,特征名和特征值,特征值和特征值之间以":"分割,分割后的长度应该等于len(itemIds) + 1</td></tr>
    <tr><td>getRequest()</td><td>返回build好的EasyRecPredictProtos.PBRequest</td></tr>
  </table>

- 验证特征一致性

  ```java
  // 获取FG之后的特征，以便和离线的特征对比一致性
  // 将DebugLevel设置成1，即可返回生成的特征
  easyrecRequest.setDebugLevel(1);
  PredictProtos.PBResponse response = client.predict(easyrecRequest);
  Map<String, String> genFeas = response.getGenerateFeaturesMap();
  for(String itemId: genFeas.keySet()) {
      System.out.println(itemId);
      System.out.println(genFeas.get(itemId));
  }
  ```

- setDebugLevel: 设置调试标志, 方便排查问题, 参数的取值范围如下:

  - 0: 仅返回预测结果, 不返回调试信息
  - 1: 只返回FG之后特征的值, 格式为key:value格式, 不返回预测结果
  - 2: 返回预测结果和FG之后的特征值
  - 3: 返回FG之后特征值, 格式为表格格式, 特征之间用\\u0002分隔, 适用于ODL[实时样本](./odl_sample.md)构造的场景
  - 4: 将FG之后生成的特征值直接写入datahub, 不通过rpc返回, 适用于ODL[实时样本](./odl_sample.md)构造的场景
  - 100: 保存请求到模型目录下, 同时返回预测结果
  - 101: 保存timeline
  - 102: 适用于召回模型, 返回user向量和Faiss检索结果

- 注意: 生产环境调用的时候设置debug_level=0，否则会导致rt上升, qps下降.
