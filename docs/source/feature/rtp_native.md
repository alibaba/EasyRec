# RTP部署

本文档介绍将EasyRec模型部署到RTP（Real Time Prediction，实时打分服务）上的流程.

- RTP目前仅支持checkpoint形式的模型部署，因此需要将EasyRec模型导出为checkpoint形式

#### 编写RTP特征配置 [fg.json](https://easyrec.oss-cn-beijing.aliyuncs.com/rtp_fg/fg.json)

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
  ],
  "reserve_default": false
}
```

- Feature配置说明：

  - [IdFeature](http://easyrec.oss-cn-beijing.aliyuncs.com/fg_docs/IdFeature.pdf)

    - is_multi: id_feature是否是多值属性
      - 默认是false, 转换成EasyRec的config时会转成IdFeature
      - 如果设成true, 转换成EasyRec的config时会转成TagFeature.
      - 多值分隔符使用chr(29)\[ctrl+v ctrl+\].
    - num_buckets: 当输入是unsigned int类型的时候，并且输入有界的时候，可以指定num_bucket为输入的最大值.
    - hash_bucket_size: 对应EasyRec feature_config.features的hash_bucket_size。hash_bucket方式是目前RTP唯一支持的embedding方式
    - embedding_dimension/embedding_dim: 对应EasyRec feature_config.features里面的embedding_dim.

  - [RawFeature](http://easyrec.oss-cn-beijing.aliyuncs.com/fg_docs/RawFeature.pdf)

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
      - 该选项对生成EasyRec config也有影响.

  - [ComboFeature](http://easyrec.oss-cn-beijing.aliyuncs.com/fg_docs/ComboFeature.pdf)

    - 需要设置embedding_dimension和hash_bucket_size.
      方法一：在fg中生成combo特征，见[ComboFeature](http://easyrec.oss-cn-beijing.aliyuncs.com/fg_docs/ComboFeature.pdf)

    ```
    {"expression": "user:user_id", "feature_name": "user_id", "feature_type":"id_feature", "value_type":"String", "combiner":"mean", "hash_bucket_size": 100000, "embedding_dim": 16, "group":"user"},
    {"expression": "user:occupation", "feature_name": "occupation", "feature_type":"id_feature", "value_type":"String", "combiner":"mean", "hash_bucket_size": 10, "embedding_dim": 16, "group":"user"},
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

    - 最终会生成两列数据（user_id和occupation），config中生成三个特征配置，分别是user_id，occupation，combo\_\_occupation_age_level.
    - final_feature_name: 该combo特征的名字.
    - feature_names: 除当前特征外，参与combo的特征，至少一项.
    - combiner, hash_bucket_size, embedding_dim 配置与上述一致.

  - [LookupFeature](http://easyrec.oss-cn-beijing.aliyuncs.com/fg_docs/LookupFeature.pdf)

    - 根据id查找对应的value.

  - [MatchFeature](http://easyrec.oss-cn-beijing.aliyuncs.com/fg_docs/MatchFeature.pdf)

    - 双层查找, 根据category和item_id查找value.
    - match Feature里面多值分隔符可以使用chr(29) (ctrl+v ctrl+\])或者逗号\[,\]， 如:

    ```
      50011740^107287172:0.2^]36806676:0.3^]122572685:0.5|50006842^16788816:0.1^]10122:0.2^]29889:0.3^]30068:19
    ```

    - needWeighting: 生成特征权重，即kv格式, kv之间用\[ctrl+v ctrl+e\]分割, 转换成TagFeature.

  - [OverLapFeature](http://easyrec.oss-cn-beijing.aliyuncs.com/fg_docs/OverLapFeature.pdf)

  - 所有feature都需要的字段:

    - group: feature所属的分组
      - 对于WideAndDeep/DeepFM是wide/deep.
      - 对于MultiTower可以自定义分组名称，如user/item/combo.
    - combiner: 默认是mean, 也可以是sum.
      - 影响数据生成和 EasyRec feature_config 生成, 主要是多值Feature.
    - [多值类型说明](http://easyrec.oss-cn-beijing.aliyuncs.com/fg_docs/%E5%A4%9A%E5%80%BC%E7%B1%BB%E5%9E%8B.pdf)
      - 多值feature使用chr(29)\[ctrl+v ctrl+\]\]作为分隔符.

- 全局配置说明:

  - reserves: 要在最终表里面要保存的字段，通常包括label, user_id, item_id等

#### 生成样本

请参见RTP文档规范，用你喜欢的方式生成样本。

举例：

准备 fg.json 配置：

```json
{
  "features": [
    {"expression": "user:user_id", "feature_name": "user_id", "feature_type":"id_feature", "value_type":"String", "combiner":"mean", "hash_bucket_size": 100000, "embedding_dim": 16, "group":"user"},
    {"expression": "user:cms_segid", "feature_name": "cms_segid", "feature_type":"id_feature", "value_type":"String", "combiner":"mean", "hash_bucket_size": 100, "embedding_dim": 16, "group":"user"},
    ...
    {"expression": "item:price", "feature_name": "price", "feature_type":"raw_feature", "value_type":"Integer", "combiner":"mean", "group":"item"},
    {"expression": "item:item_id", "feature_name": "item_id", "feature_type":"id_feature", "value_type":"String", "combiner":"mean", "hash_bucket_size": 100000, "embedding_dim": 16, "group":"item"},
    {"expression": "user:tag_category_list", "feature_name": "user_tag_cate", "feature_type":"id_feature", "hash_bucket_size":100000, "group":"user"},
    {"map": "user:tag_brand_list", "key":"item:brand", "feature_name": "combo_brand", "feature_type":"lookup_feature",  "needDiscrete":true, "hash_bucket_size":100000, "group":"combo"},
    {"map": "user:tag_category_list", "key":"item:cate_id", "feature_name": "combo_cate_id", "feature_type":"lookup_feature",  "needDiscrete":true, "hash_bucket_size":10000, "group":"combo"}
  ],
  "reserves": [
    "user_id", "item_id", "clk"
  ],
  "reserve_default": false
}
```

准备数据，例如：

| clk | user_id | item_id | tag_category_list | price | age_level | ... |
| --- | ------- | ------- | ----------------- | ----- | --------- | --- |
| 1   | 122017  | 389957  | 4589              | 10    | 0         | ... |

下载 fg_on_odps 的 jar包 [fg_on_odps_nodep_5u_file20-1.4.20-jar-with-dependencies.jar](https://easyrec.oss-cn-beijing.aliyuncs.com/deploy/fg_on_odps_nodep_5u_file20-1.4.20-jar-with-dependencies.jar)

生成样本：

```sql
add jar fg_on_odps_nodep_5u_file20-1.4.20-jar-with-dependencies.jar -f;
add file fg.json -f;

set odps.sql.planner.mode=sql;
set odps.isolation.session.enable=true;
set odps.sql.counters.dynamic.limit=true;

drop table if exists dssm_taobao_fg_train_out;
create table dssm_taobao_fg_train_out(clk bigint, user_id string, item_id string, features string);
jar -libjars fg_on_odps_nodep_5u_file20-1.4.20-jar-with-dependencies.jar
    -resources fg.json
    -classpath fg_on_odps_nodep_5u_file20-1.4.20-jar-with-dependencies.jar com.taobao.fg_on_odps.FGMapperTF
	dssm_test_feature_table
    dssm_taobao_fg_train_out
    fg.json;

drop table if exists dssm_taobao_fg_test_out;
create table dssm_taobao_fg_test_out(clk bigint, user_id string, item_id string, features string);
jar -libjars fg_on_odps_nodep_5u_file20-1.4.20-jar-with-dependencies.jar
    -resources fg.json
    -classpath fg_on_odps_nodep_5u_file20-1.4.20-jar-with-dependencies.jar com.taobao.fg_on_odps.FGMapperTF
	dssm_test_feature_table_test
    dssm_taobao_fg_test_out
    fg.json;

--下载查看数据(可选)
tunnel download dssm_taobao_fg_test_out dssm_taobao_fg_test_out.txt -fd=';';
```

这会生成这样的样本表：

| clk | user_id | item_id | features                                              |
| --- | ------- | ------- | ----------------------------------------------------- |
| 1   | 122017  | 389957  | tag_category_list^C4589^Bprice^C10^Bage_level^C0^B... |

#### 编写EasyRec配置 fg.config

示例

```proto
model_dir: "oss://easyrec/rtp_fg_demo"

train_config {
  optimizer_config {
    use_moving_average: false
    adam_optimizer {
      learning_rate {
        exponential_decay_learning_rate {
          initial_learning_rate: 0.0001
          decay_steps: 100000
          decay_factor: 0.5
          min_learning_rate: 1e-07
        }
      }
    }
  }
  num_steps: 1000
  sync_replicas: false
  log_step_count_steps: 200
}

fg_json_path: "oss://easyrec/rtp_fg/fg.json"

data_config {
  batch_size: 1024
  label_fields: "clk"
  input_type: OdpsRTPInputV2
  separator: ""
  selected_cols: "clk,features"
  rtp_separator: ";"
}

model_config:{
  model_class: "DeepFM"
  feature_groups: {
    group_name: 'deep'
    feature_names: 'user_id'
    feature_names: 'cms_segid'
    ...
    feature_names: 'brand'
    wide_deep:DEEP
  }
  feature_groups: {
    group_name: 'wide'
    feature_names: 'user_id'
    feature_names: 'cms_segid'
    ...
    feature_names: 'brand'
    wide_deep:WIDE
  }
  deepfm {
    wide_output_dim: 16
    dnn {
      hidden_units: [128, 64, 32]
    }
    final_dnn {
      hidden_units: [128, 64]
    }
    l2_regularization: 1e-6
  }
  embedding_regularization: 5e-5
}

export_config {
  multi_placeholder: false
  export_rtp_outputs: true
}
```

- fg_json_path: RTP FG 配置文件即 fg.json 的路径
- data_config
  - input_fields: 输入字段配置，无需设置，EasyRec 会根据 fg.json 自动生成
  - input_type: 须填写`OdpsRTPInputV2`，仅此输入类型与RTP在线服务兼容
- feature_configs: 特征配置，无需设置，EasyRec 会根据 fg.json 自动生成
- export_config
  - export_rtp_outputs: 须设置为`true`，令 EasyRec 在输出图中加入 RTP 预测节点

#### 启动训练

- 上传fg.config和fg.json到oss
- 启动训练

```sql
pai -name easy_rec_ext -project algo_public
-Dconfig=oss://bucket-name/easy_rec_test/fg.config
-Dcmd=train
-Dtables='odps://project-name/tables/dssm_taobao_fg_train_out,odps://project-name/tables/dssm_taobao_fg_test_out'
-Dcluster='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":3, "cpu":1000, "gpu":100, "memory":40000}}'
-Darn='acs:ram::xxx:role/ev-ext-test-oss'
-Dbuckets='oss://bucket-name/'
-DossHost='oss-cn-xxx.aliyuncs.com'
-Dmodel_dir='oss://bucket-name/easy_rect_test_model/202203031730/data'
-Dselected_cols='clk,features';
```

#### 模型导出

```sql
pai -name easy_rec_ext -project algo_public
-Dconfig='oss://bucket-name/easy_rec_test/fg.config'
-Dcmd=export_checkpoint
-Dcluster='{"worker" : {"count":1, "cpu":1000, "gpu":0, "memory":40000}}'
-Darn='acs:ram::xxx:role/ev-ext-test-oss'
-Dbuckets='oss://bucket-name/'
-DossHost='oss-cn-xxx.aliyuncs.com'
-Dmodel_dir='oss://bucket-name/easy_rect_test_model/202203031730/data'
-Dexport_dir='oss://bucket-name/easy_rect_test_model_export/202203031730/data'
-Dselected_cols='clk,features'
-Dbatch_size=256;
```

- 在`export_dir`指定的目录下生成名为 model.ckpt.\* 的checkpoint文件，将`export_dir`指定的目录所对应的主目录指定为RTP模型表的模型目录即可, 例如:
- `export_dir`为 oss://bucket-name/easy_rect_test_model_export/202203031730/data，则将RTP模型目录指定为 oss://bucket-name/easy_rect_test_model_export
- Note: 弹内pai版本ossHost, arn, buckets参数指定方式和公有云版本有差异，具体见[使用文档](../quick_start/mc_tutorial_inner.md) RTP Serving部分.

#### EAS部署

- 还可以将模型部署到EAS上，参考[文档](../predict/%E5%9C%A8%E7%BA%BF%E9%A2%84%E6%B5%8B).
