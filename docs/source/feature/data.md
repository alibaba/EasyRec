# 数据格式

EasyRec作为阿里云PAI的推荐算法包，可以无缝对接MaxCompute的数据表，也可以读取OSS中的大文件，还支持E-MapReduce环境中的HDFS文件，也支持local环境中的csv文件。

为了识别这些输入数据中的字段信息，需要设置相应的字段名称和字段类型、设置默认值，帮助EasyRec去读取相应的数据。设置label字段，作为训练的目标。为了适配多目标模型，label字段可设置多个。

另外还有一些参数如prefetch_size，是tensorflow中读取数据需要设置的参数。

## 一个最简单的data config的配置

这个配置里面，只有三个字段，用户ID（uid）、物品ID（item_id）、label字段（click）。

OdpsInputV2表示读取MaxCompute的表作为输入数据。如果是本地机器上训练，注意使用CSVInput类型。

```protobuf
data_config {
  batch_size: 2048
  input_fields {
    input_name: "click"
    input_type: INT32
  }
  input_fields {
    input_name: "uid"
    input_type: STRING
  }
  input_fields {
    input_name: "item_id"
    input_type: STRING
  }
  label_fields: "click"
  num_epochs: 1
  prefetch_size: 32
  input_type: OdpsInputV2
}

```

## input_fields:

input_fields字段:

- input_name，方便在后续的 feature_config.featurs 中和 data_config.label_fields 中引用;
- input_type，默认是STRING，可以不设置。可选的字段参考[DatasetConfig.FieldType](../proto.html)
- default_val，默认是空，**注意默认值都是设置成字符串**
  - 如果input是INT32类型，并且默认值是6，那么default_val是"6";
  - 如果input是DOUBLE类型，并且默认值是0.5，那么default_val是"0.5";
- input_dim, 目前仅适用于RawFeature类型，可以指定多维数据，如一个图片的embedding vector.
- user_define_fn, 目前仅适用于label，指定用户自定义函数名，以对label进行处理.
- user_define_fn_path, 如需引入oss/hdfs上的用户自定义函数，需指定函数路径.
- user_define_fn_res_type, 指定用户自定义函数的输出值类型.

```protobuf
  input_fields: {
    input_name: "label"
    input_type: DOUBLE
    default_val:"0"
  }
```

```protobuf
  input_fields {
    input_name:'clk'
    input_type: DOUBLE
    user_define_fn: 'tf.math.log1p'
  }
```

```protobuf
  input_fields {
    input_name:'clk'
    input_type: INT64
    user_define_fn: 'remap_lbl'
    user_define_fn_path: 'samples/demo_script/process_lbl.py'
    user_define_fn_res_type: INT64
  }
```

process_lbl.py:

```python
import numpy as np
def remap_lbl(labels):
    res = np.where(labels<5, 0, 1)
    return res
```

- **注意:**
  - input_fields的顺序和odps table里面字段的顺序不需要保证一一对应的
  - input_fields和csv文件里面字段的顺序必须是一一对应的(csv文件没有header)
  - input_fields里面input_type必须和odps table/csv文件对应列的类型一致
  - maxcompute上不建议使用FLOAT类型，建议使用DOUBLE类型

### input_type:

目前支持一下几种[input_type](../proto.html#protos.DatasetConfig.InputType):

- CSVInput，表示数据格式是CSV，注意要配合separator使用

  - 需要指定train_input_path和eval_input_path

  ```protobuf
  train_input_path: "data/test/dwd_avazu_ctr_train.csv"
  eval_input_path: "data/test/dwd_avazu_ctr_test.csv"
  ```

- OdpsInputV2，如果在MaxCompute上运行EasyRec, 则使用OdpsInputV2

  - 需要指定train_input_path和eval_input_path
  - 可以通过pai命令传入, [参考](../train.md#on-pai)

- OdpsInputV3, 如果在本地或者[DataScience](https://help.aliyun.com/document_detail/170836.html)上访问MaxCompute Table, 则使用OdpsInputV3

- HiveInput和HiveParquetInput, 在Hadoop集群上访问Hive表

  - 需要配置hive_train_input和hive_eval_input
  - 参考[HiveConfig](../proto.html#protos.HiveConfig)

  ```protobuf
  hive_train_input {
    host: "192.168.1"
    username: "admin"
    table_name: "census_income_train_simple"
  }
  hive_eval_input {
    host: "192.168.1"
    username: "admin"
    table_name: "census_income_eval_simple"
  }
  ```

- 如果需要使用RTP FG, 那么：

  - 在EMR或者本地运行EasyRec，应使用RTPInput或者HiveRTPInput;
  - 在Odps上运行，则应使用OdpsRTPInput

- KafkaInput & DatahubInput: [实时训练](../online_train.md)需要用到的input类型

  - KafkaInput需要配置kafka_train_input 和 kafka_eval_input
    - 参考[KafkaServer](../proto.html#protos.KafkaServer)
  - DatahubServer需要配置datahub_train_input 和 datahub_eval_input
    - 参考[DataHubServer](../proto.html#protos.DatahubServer)

### separator:

- 使用csv格式的输入需要指定separator作为列之间的分隔符
- 默认是半角逗号","
- 可使用不可见字符作为分隔符（二进制分隔符），如'\\001', '\\002'等

### label_fields

- label相关的列名，至少设置一个，可以根据算法需要设置多个，如多目标算法

  ```protobuf
    label_fields: "click"
    label_fields: "buy"
  ```

- 列名必须在data_config中出现过

### prefetch_size

- data prefetch，以batch为单位，默认是32
- 设置prefetch size可以提高数据加载的速度，防止数据瓶颈。但是当batchsize较小的时候，该值可适当调小。

### shard && file_shard

- shard按sample粒度对数据集进行分片
- file_shard按文件粒度对数据集进行分片
  - 适用于输入由很多小文件组成的场景
  - 不适用于maxcompute table数据源

### shuffle

- 默认值是true，不做shuffle则设置为false
- 设置shuffle，可以对训练数据进行shuffle，获得更好的效果
- 如果有多个输入文件，文件之间也会进行shuffle

### shuffle_buffer_size

- 默认值32
- shuffle queue的大小，代表每次shuffle的样本数
- 越大训练效果越好, 但是内存消耗也会变大
- 通常建议在训练前做一次[全局shuffle](../optimize.md#3shuffle)，训练过程中使用比较小的buffer_size进行shuffle或者不再shuffle

### 更多配置

- [参考文档](https://easyrec.readthedocs.io/en/latest/proto.html#easy_rec%2fpython%2fprotos%2fdataset.proto)
