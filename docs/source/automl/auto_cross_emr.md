# AUTO-CROSS(EMR)

### 输入数据:

输入一般是csv格式的文件。 如下所示，列之间用,分割

- 示例数据（小数据集）:
  - train: [ctr_train.csv](https://easyrec.oss-cn-beijing.aliyuncs.com/data/autocross/ctr_train.csv)
  - test: [ctr_test.csv](https://easyrec.oss-cn-beijing.aliyuncs.com/data/autocross/ctr_test.csv)
  - 数据示例:

```
1,10,1005,0,85f751fd,c4e18dd6,50e219e0,0e8e4642,b408d42a,09481d60,a99f214a,5deb445a, f4fffcd0,1,0,2098,32,5,238,0,56,0,5
```

- Copy data to HDFS:

```bash
hadoop fs -mkdir -p hdfs:///user/fe/data/
hadoop fs -put ctr_train.csv hdfs:///user/fe/data/
hadoop fs -put ctr_test.csv hdfs:///user/fe/data/
```

### AutoCross

- AutoCross yaml配置文件：[ctr_autocross.yaml](https://easyrec.oss-cn-beijing.aliyuncs.com/data/autocross/ctr_autocross.yaml)
- alink环境配置文件，另存为a[link.env](https://easyrec.oss-cn-beijing.aliyuncs.com/data/autocross/alink.env)

```bash
userId=default
alinkServerEndpoint=http://localhost:9301
hadoopHome=/usr/lib/hadoop-current
hadoopUserName=hadoop
token=ZSHTIeEkwrtZJJsN1ZZmCJJmr5jaj1wO
```

- 使用 pai-automl-fe 提交任务

```bash
pai-automl-fe run -e configs/alink.env --config configs/ctr_autocross.yaml --mode emr
```

### 对接EasyRec

EasyRec使用请参考文档 [EMR Train](../train.md)。
以下说明AutoCross后的数据对接easy_rec的配置（[ctr_deepmodel_ac.config](https://easyrec.oss-cn-beijing.aliyuncs.com/data/autocross/ctr_deepmodel_ac.config)）

#### 数据据相关

```protobuf
# 数据相关的描述
data_config {
  separator: ","
  input_fields: {
    input_name: "label"
    input_type: FLOAT
    default_val:""
  }
  input_fields: {
    input_name: "hour"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c1"
    input_type: STRING
    default_val:""
  }
  ...
  # 以下是新增加的
  input_fields: {
    input_name: "c0_all"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c1_c"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c2_c"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c3_c"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c21"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c13_c"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c14_c"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c15_c"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c16_c"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c17_c"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c18_c"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c19_c"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c20_c"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "c21_c"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "cross_1"
    input_type: STRING
    default_val:""
  }
  input_fields: {
    input_name: "cross_2"
    input_type: STRING
    default_val:""
  }
```

#### 特征相关

```protobuf
feature_config: {
  features: {
    input_names: "hour"
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 50
  }
  features: {
    input_names: "c1"
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 10
  }
  ...
  # 以下新增加的交叉列
  features: {
    input_names: "cross_1"
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 500
  }
  features: {
    input_names: "cross_2"
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 500
  }
}
```

#### 模型相关

```protobuf
model_config:{
  model_class: "MultiTower"
  feature_groups: {
  ...
  }
  feature_groups: {
    ...
  }
  feature_groups: {
    group_name: "user_item"
    feature_names: "hour"
    feature_names: "c14"
    feature_names: "c15"
    feature_names: "c16"
    feature_names: "c17"
    feature_names: "c18"
    feature_names: "c19"
    feature_names: "c20"
    feature_names: "c21"
    feature_names: "cross_1"
    feature_names: "cross_2"
    wide_deep:DEEP
  }
```

使用el_submit提交训练即可，请参照 [EMR Train](../train.md)。
