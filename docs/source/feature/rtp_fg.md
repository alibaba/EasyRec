# RTP FG

- RTP FG能够以比较高的效率生成一些复杂的特征，如MatchFeature和LookupFeature, 线上线下使用同一套代码保证一致性.

- 其生成的特征可以接入EasyRec进行训练，从RTP FG的配置(fg.json)可以生成EasyRec的配置文件(pipeline.config).

- 线上部署的时候提供带FG功能的EAS processor，一键部署.

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
 ],
 "multi_val_sep": "|"
}
```

- Feature配置说明：

  - [IdFeature](http://easyrec.oss-cn-beijing.aliyuncs.com/fg_docs/IdFeature.pdf)

    - is_multi: id_feature是否是多值属性

      - 默认是false, 转换成EasyRec的config时会转成IdFeature

      - 如果设成true, 转换成EasyRec的config时会转成TagFeature.

      - 多值分隔符使用chr(29)\[ctrl+v ctrl+\].

    - vocab_file: 词典文件路径，根据词典将对应的输入映射成ID.

    - vocab_list: 词典list，根据词典将对应的输入映射成ID.

    - num_buckets: 当输入是unsigned int类型的时候，并且输入有界的时候，可以指定num_bucket为输入的最大值.

    - hash_bucket_size: 对应EasyRec feature_config.features的hash_bucket_size.

      - 和vocab_file, vocab_list相比，优势是不需要词典，词典可以是不固定的.

      - 劣势是需要设置的容量比较大，容易导致hash冲突.

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

  - separator: sparse格式里面，特征之间的分隔符，不指定默认是","，

    - 训练时，对稠密格式没有影响，对稀疏格式有影响
    - 预测时，item feature在redis里面存储的是稀疏格式，因此是有影响的

    ```
    i_item_id:10539078362,i_seller_id:21776327,...
    ```

  - multi_val_sep: 多值特征的分隔符，不指定默认是chr(29) 即"\\u001D"

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

```sql
-- taobao_train_input.txt oss://easyrec/data/rtp/
-- wget http://easyrec.oss-cn-beijing.aliyuncs.com/data/rtp/taobao_train_input.txt
-- wget http://easyrec.oss-cn-beijing.aliyuncs.com/data/rtp/taobao_test_input.txt
drop table if exists taobao_train_input;
create table if not exists taobao_train_input(`label` BIGINT,user_id STRING,item_id STRING,context_feature STRING,user_feature STRING,item_feature STRING);
tunnel upload taobao_train_input.txt taobao_train_input -fd=';';
drop table if exists taobao_test_input;
create table if not exists taobao_test_input(`label` BIGINT,user_id STRING,item_id STRING,context_feature STRING,user_feature STRING,item_feature STRING);
tunnel upload taobao_test_input.txt taobao_test_input -fd=';';
```

- 稠密格式的数据，每个特征是单独的一列，如：

| label | user_id | item_id | tag_category_list | new_user_class_level | age_level |
| ----- | ------- | ------- | ----------------- | -------------------- | --------- |
| 1     | 122017  | 389957  | 4589              |                      | 0         |

```sql
  drop table if exists taobao_train_input;
  create table taobao_train_input_dense(label bigint, user_id string, item_id string, tag_category_list bigint, ...);
```

- **Note:** 特征列名可以加上prefix: **"user\_\_", "item\_\_", "context\_\_"**

```
  如: 列名ctx_position也可以写成 context__ctx_position
```

#### 生成样本

- 下载rtp_fg [jar ](https://easyrec.oss-cn-beijing.aliyuncs.com/deploy/fg_on_odps-1.3.57-jar-with-dependencies.jar)包
- 生成特征

```sql
add jar target/fg_on_odps-1.3.57-jar-with-dependencies.jar -f;
add file fg.json -f;

set odps.sql.planner.mode=sql;
set odps.isolation.session.enable=true;
set odps.sql.counters.dynamic.limit=true;

drop table if exists taobao_fg_train_out;
create table taobao_fg_train_out(label bigint, user_id string, item_id string,  features string);
jar -resources fg_on_odps-1.3.57-jar-with-dependencies.jar,fg.json -classpath fg_on_odps-1.3.57-jar-with-dependencies.jar com.taobao.fg_on_odps.EasyRecFGMapper -i taobao_train_input -o taobao_fg_train_out -f fg.json;
drop table if exists taobao_fg_test_out;
create table taobao_fg_test_out(label bigint, user_id string, item_id string,  features string);
jar -resources fg_on_odps-1.3.57-jar-with-dependencies.jar,fg.json -classpath fg_on_odps-1.3.57-jar-with-dependencies.jar com.taobao.fg_on_odps.EasyRecFGMapper -i taobao_test_input -o taobao_fg_test_out -f fg.json;

--下载查看数据(可选)
tunnel download taobao_fg_test_out taobao_fg_test_out.txt -fd=';';
```

- EasyRecFGMapper参数格式:
  - -i, 输入表
    - 支持分区表，分区表可以指定partition，也可以不指定partition，不指定partition时使用所有partition
    - **分区格式示例:** my_table/day=20201010,sex=male
    - 可以用多个-i指定**多个表的多个分区**
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

- --separator: feature之间的分隔符, 默认是CTRL_B(\\u0002)

- --selected_cols: 指定输入列，包括label、\[sample_weight\]和features，其中label可以指定多列，表示要使用多个label(一般是多任务模型),  最后一列必须是features, 如:

  ```
  label0,label1,sample_weight,features
  ```

  - 注意不要有**空格**，其中 sample_weight 列是可选的，可以没有

- --incol_separator: feature内部的分隔符，即多值分隔符，默认是CTRL_C(\\u0003)

- --input_type: 输入类型，默认是OdpsRTPInput, 如果在EMR上使用或者本地使用，应该用RTPInput, 如果使用RTPInput那么--selected_cols也需要进行修改, 使用对应的列的id:

  ```
  0,4
  ```

  - 其中第0列是label, 第4列是features
  - 还需要指定--rtp_separator，表示label和features之间的分隔符, 默认是";"

- --train_input_path, 训练数据路径

  - MaxCompute上不用指定，在训练的时候指定

- --eval_input_path, 评估数据路径

  - MaxCompute上不用指定，在训练的时候指定

#### 启动训练

- 上传fg.config到oss
- 启动训练

```sql
pai -name easy_rec_ext
-Dconfig=oss://bucket-name/easy_rec_test/fg.config
-Dcmd=train
-Dtables='odps://project-name/tables/taobao_fg_train_out,odps://project-name/tables/taobao_fg_test_out'
-Dcluster='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":3, "cpu":1000, "gpu":100, "memory":40000}}'
-Darn=acs:ram::xxx:role/ev-ext-test-oss
-Dbuckets=oss://bucket-name/
-DossHost=oss-cn-xxx.aliyuncs.com
-Deval_method=separate;
```

环境里没有安装easy_rec_ext ，则上传easy_rec.tar.gz包

```sql
pai -name tensorflow1120_cpu_ext
    -Dscript='oss://<path>/easy_rec.tar.gz'
    -DentryFile='run.py'
    -Dbuckets='oss://<bucket-name>/'
    -Dtables='odps://<project-name>/tables/<train_table_name>/dt=${bizdate},odps://<project-name>/tables/<test_table_name>/dt=${bizdate}'
    -Darn='acs:ram::xxx:role/aliyunodpspaidefaultrole'
    -DossHost='oss-us-west-1-internal.aliyuncs.com'
    -Dcluster='{
      \"ps\": {
          \"count\" : 4,
          \"cpu\" : 600,
          \"memory\" : 30000
      },
      \"worker\" : {
          \"count\" : 33,
          \"cpu\" : 800,
          \"memory\" : 30000
      }
    }'
    -DuserDefinedParameters='--cmd train --config oss://<path>/fg.config --model_dir oss://<model_path>/ --train_tables odps://<project-name>/tables/<train_table_name>/dt=${bizdate} --eval_tables odps://<project-name>/tables/<test_table_name>/dt=${bizdate} --with_evaluator'
;
```

#### 模型导出

```sql
pai -name tensorflow1120_cpu_ext
    -Dscript='oss://<path>/easy_rec.tar.gz'
    -DentryFile='run.py'
    -Dbuckets='oss://<bucket-name>/'
    -Darn='acs:ram::xxx:role/aliyunodpspaidefaultrole'
    -DossHost='oss-us-west-1-internal.aliyuncs.com'
    -DuserDefinedParameters='--cmd export --config=oss://<model_path>/pipeline.config --export_dir=oss://<export_path>/ --asset_files=oss://<path>/fg.json';
;

```

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
  jar -resources fg_on_odps-1.3.57-jar-with-dependencies.jar,incre_fg.json -classpath fg_on_odps-1.3.57-jar-with-dependencies.jar com.taobao.fg_on_odps.EasyRecFGMapper -i taobao_test_input -o taobao_fg_test_out -f incre_fg.json;
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
  "name":"easyrec_processor",
  "baseimage": "registry.cn-shanghai.aliyuncs.com/eas/eas-worker-amd64:0.4.22",
  "metadata": {
    "region": "us-west-1",
    "cpu": 6,
    "memory": 20000,
    "instance": 3
  },
  "model_config":"{\"holo-conf\":{\"url\":\"postgresql://<AccessKeyID>:<AccessKeySecret>@<域名>:<port>/<database>\",\"prefix\":\"fg_*\",\"table\" : [{\"name\": \"<schema>.<table_name>\",\"key\" : \"<index_column_name>\",\"value\": \"<column_name>\",\"period\": 2880}]},\"period\": 2880,\"fg\":true,\"multitargets\":true,\"outputs\":\"probs_ctr,probs_cvr\",\"inter_op_parallelism_threads\": 6, \"intra_op_parallelism_threads\": 6, \"fg_ins_num\":2}",
  "model_path": "oss://<model_path>/",
  "processor_path": "oss://easyrec/deploy/processor/easyrec_holo_broadwell.tar.gz",
  "processor_entry": "libtf_predictor.so",
  "token": "Y2E4OGY2MTBkODFhMzJhMDUzODM0YmE4OGRjZTI2MTgxYWNhOWRkNw==",
  "processor_type": "cpp"
}

EOF
# 执行部署命令。
#/home/admin/usertools/tools/eascmd -i <AccessKeyID>  -k  <AccessKeySecret>   -e pai-eas.us-west-1.aliyuncs.com create echo.json
/home/admin/usertools/tools/eascmd -i <AccessKeyID>  -k  <AccessKeySecret>   -e pai-eas.us-west-1.aliyuncs.com update easyrec_processor -s echo.json


```

训练导出的时候需要修改fg.config ，保证导出的模型是支持多个place_holder 的输入

```
export_config {
  multi_placeholder: true
}
```

- processor_path， processor_entry， processor_type 自定义 easyrec processor  设置，与示例保持一致即可

- model_config: eas 部署配置。主要控制把 item 特征加载到内存中。目前数据源支持redis和holo

  - redis-conf: 配置redis 访问的相关配置，包括 url, password
    - prefix: item_id key的前缀, 为了和其它的key(如user_id等)区分开来
    - cluster: cluster模式访问redis, 默认是false, 使用单例模式
  - pool_size: redis connection pool size
  - period: item feature reload period, 单位minutes

- 更多选项:

  - model_config:
    - fg_ins_num: fg并行数，可以加快fg的计算速度
    - multitargets: 是否多目标模型
    - outputs: saved_model output signatures, 如果有多个，之间用,分割

  ```
    "model_config":{
      "fg_ins_num": 4,
      "multitargets": true,
      "outputs": "probs_ctr,probs_cvr",
      ...
    }
  ```

- holo-conf: 也支持使用[holo](https://www.aliyun.com/product/bigdata/hologram)存储item feature, 好处是支持增量更新

  - 需要创建一张holo表, 包含3列:
    ```
    ｜item_id｜item_features｜update_time｜
    ```
  - url: holo url
  - user: holo db username
  - password: holo db password
  - dbname: holo dbname
  - table: holo table name
  - key: name of the column store item_ids
  - value: name of the column store item features

```
  {
    "model_config":{
      "holo-conf":{
        "url":"hgprecn-cn-09k22ikm5008-cn-hangzhou.hologres.aliyuncs.com",
        "user":"admin",
        "password":"1234567",
        "dbname":"easyrec_test",
        "table":"test_table",
        "key":"item_id",
        "value":"item_features"
      }
      ...
    }
  }
```

#### 客户端访问

同eas sdk 中的TFRequest类似，easyrec 也是使用ProtoBuffer 作为传输协议. proto 文件定义：

```protobuf
syntax = "proto3";

option go_package = ".;easyrec";
option java_package = "com.alibaba.pairec.processor.proto";
option java_outer_classname = "PredictProtos";

enum ArrayDataType {
  // Not a legal value for DataType. Used to indicate a DataType field
  // has not been set.
  DT_INVALID = 0;
  // Data types that all computation devices are expected to be
  // capable to support.
  DT_FLOAT = 1;
  DT_DOUBLE = 2;
  DT_INT32 = 3;
  DT_UINT8 = 4;
  DT_INT16 = 5;
  DT_INT8 = 6;
  DT_STRING = 7;
  DT_COMPLEX64 = 8;  // Single-precision complex
  DT_INT64 = 9;
  DT_BOOL = 10;
  DT_QINT8 = 11;     // Quantized int8
  DT_QUINT8 = 12;    // Quantized uint8
  DT_QINT32 = 13;    // Quantized int32
  DT_BFLOAT16 = 14;  // Float32 truncated to 16 bits.  Only for cast ops.
  DT_QINT16 = 15;    // Quantized int16
  DT_QUINT16 = 16;   // Quantized uint16
  DT_UINT16 = 17;
  DT_COMPLEX128 = 18;  // Double-precision complex
  DT_HALF = 19;
  DT_RESOURCE = 20;
  DT_VARIANT = 21;  // Arbitrary C++ data types
}
// Dimensions of an array
message ArrayShape {
  repeated int64 dim = 1 [packed = true];
}
// Protocol buffer representing an array
message ArrayProto {
  // Data Type.
  ArrayDataType dtype = 1;
  // Shape of the array.
  ArrayShape array_shape = 2;
  // DT_FLOAT.
  repeated float float_val = 3 [packed = true];
  // DT_DOUBLE.
  repeated double double_val = 4 [packed = true];
  // DT_INT32, DT_INT16, DT_INT8, DT_UINT8.
  repeated int32 int_val = 5 [packed = true];
  // DT_STRING.
  repeated bytes string_val = 6;
  // DT_INT64.
  repeated int64 int64_val = 7 [packed = true];
  // DT_BOOL.
  repeated bool bool_val = 8 [packed = true];
}
// context features
message ContextFeatures {
  repeated PBFeature features = 1;
}

message PBFeature {
  oneof value {
    int32 int_feature = 1;
    int64 long_feature = 2;
    string string_feature = 3;
    float float_feature = 4;
  }
}

// PBRequest specifies the request for aggregator
message PBRequest {
  // debug mode
  int32 debug_level = 1;

  // user features
  map<string, PBFeature> user_features = 2;

  // item ids
  repeated string item_ids = 3;

  // context features for each item
  map<string, ContextFeatures> context_features = 4;
}

// return results
message Results {
  repeated double scores = 1 [packed = true];
}

enum StatusCode {
  OK = 0;
  INPUT_EMPTY = 1;
  EXCEPTION = 2;
}

// PBResponse specifies the response for aggregator
message PBResponse {
  // results
  map<string, Results> results = 1;

  // item features
  map<string, string> item_features = 2;

  // generate features
  map<string, string> generate_features = 3;

  // context features
  map<string, ContextFeatures> context_features = 4;

  string error_msg = 5;
  
  StatusCode status_code = 6;

  // item ids
  repeated string item_ids = 7;

  repeated string outputs = 8;

  // all fg input features
  map<string, string> raw_features = 9;

  map<string, ArrayProto> tf_outputs = 10;
}
```

提供了 java 的客户端实例，[客户端 jar 包地址](http://easyrec.oss-cn-beijing.aliyuncs.com/deploy/easyrec-eas-client-0.0.2-jar-with-dependencies.jar).
下载后的 jar 通过下面命令安装到本地 mvn 库里.

```
mvn install:install-file -Dfile=easyrec-eas-client-0.0.2-jar-with-dependencies.jar -DgroupId=com.alibaba.pairec -DartifactId=easyrec-eas-client -Dversion=0.0.2 -Dpackaging=jar
```

然后在pom.xml里面加入:

```
<dependency>
    <groupId>com.alibaba.pairec</groupId>
    <artifactId>easyrec-eas-client</artifactId>
    <version>0.0.2</version>
</dependency>
```

java 客户端测试代码参考：

```java
import com.alibaba.pairec.processor.client.*;

PaiPredictClient client = new PaiPredictClient(new HttpConfig());
client.setEndpoint(cmd.getOptionValue("e"));
client.setModelName(cmd.getOptionValue("m"));

EasyrecRequest easyrecRequest = new EasyrecRequest(separator);
easyrecRequest.appendUserFeatureString(userFeatures);
easyrecRequest.appendContextFeatureString(contextFeatures);
easyrecRequest.appendItemStr(itemIdStr, ",");

PredictProtos.PBResponse response = client.predict(easyrecRequest);

for (Map.Entry<String, PredictProtos.Results> entry : response.getResultsMap().entrySet()) {
      String key = entry.getKey();
      PredictProtos.Results value = entry.getValue();
      System.out.print("key: " + key);
      for (int i = 0; i < value.getScoresCount(); i++) {
          System.out.format(" value: %.4f ", value.getScores(i));
      }
}
```

- 验证特征一致性

```
...

easyrecRequest.setDebugLevel(1);
PredictProtos.PBResponse response = this.predict(easyrecRequest);
for (Map.Entry<String, PredictProtos.Results> entry :
     response.getResultsMap().entrySet()) {
   Map<String, String> genFeas = response.getGenerateFeaturesMap();
   for(String itemId: genFeas.keySet()) {
     System.out.println(itemId);
     System.out.println(genFeas.get(itemId));
   }
}
```

- Note: 生产环境调用的时候不要设置debug，会导致rt升高，qps下降.
