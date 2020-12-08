# 数据

EasyRec作为阿里云PAI的推荐算法包，可以无缝对接MaxCompute的数据表，也可以读取OSS中的大文件，还支持E-MapReduce环境中的HDFS文件，也支持local环境中的csv文件。

为了识别这些输入数据中的字段信息，需要设置相应的字段名称和字段类型、设置默认值，帮助EasyRec去读取相应的数据。设置label字段，作为训练的目标。为了适应多目标模型，label字段可以设置多个。

另外还有一些参数如prefetch\_size，是tensorflow中读取数据需要设置的参数。

## 一个最简单的data config的配置

这个配置里面，只有三个字段，用户ID（uid）、物品ID（item\_id）、label字段（click）。

OdpsInputV2表示读取MaxCompute的表作为输入数据。

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

## input\_fields:

针对csv文件中的某一列或者odps table中的每一个字段，在data\_config都需有一个input\_fields与之对应，顺序也务必是一致的。
input\_fields字段:

- input\_name，方便在后续的feature\_configs中和data\_config.label\_fields中引用;
- input\_type，默认是STRING，可以不设置。可选的字段参考[DatasetConfig.FieldType](../proto.html)
- default\_val，默认是空，**注意默认值都是设置成字符串**
  - 如果input是INT类型，并且默认值是6，那么default\_val是"6";
  - 如果input是FLOAT类型，并且默认值是0.5，那么default\_val是"0.5";

```protobuf
  input_fields: {
    input_name: "label"
    input_type: FLOAT
    default_val:""
  }
```

- \*\*注意: \*\*
  - input\_fields的顺序和odps table里面字段的顺序必须是一一对应的:
  - input\_fields和csv文件里面字段的顺序必须是一一对应的(csv文件没有header)
  - input\_fields里面input\_type必须和odps table/csv文件对应列的类型一致，或者是可以转换的类型，如:
    - odps table里面string类型的"64"可以转成int类型的64
    - odps table里面string类型的"abc"不能转成int类型

### input\_type:

- 默认值是CSVInput，表示数据格式是CSV，注意要配合separator使用
- 如果在Odps上，应使用OdpsInputV2

#### separator:

- 使用csv格式的输入需要指定separator作为列之间的分隔符
- 默认是半角逗号","
- 可使用不可见字符作为分隔符（二进制分隔符），如'\\001', '\\002'等

#### label\_fields

- label相关的列名，至少设置一个，可以根据算法需要设置多个，如多目标算法

  ```protobuf
    label_fields: "click"
    label_fields: "buy"
  ```

- 列名必须在data\_config中出现过

### prefetch\_size

- data prefetch，以batch为单位，默认是32
- 设置prefetch size可以提高数据加载的速度，防止数据瓶颈

### shuffle

- 默认值是true，不做shuffle则设置为false
- 设置shuffle，可以对训练数据进行shuffle，获得更好的效果

### shuffle\_buffer\_size

- 默认值32
- shuffle queue的大小，代表每次shuffle的数据量
- 越大效果越好
- 建议在训练前做一次充分的彻底的shuffle
