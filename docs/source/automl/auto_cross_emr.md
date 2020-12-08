# AUTO-CROSS(EMR)

### 输入数据:

输入一般是csv格式的文件。 如下所示，列之间用,分割

- 示例数据（小数据集）:
  - train: [ctr\_train.csv](https://yuguang-test.oss-cn-beijing.aliyuncs.com/fe/data/ctr_train.csv)
  - test: [ctr\_test.csv](https://yuguang-test.oss-cn-beijing.aliyuncs.com/fe/data/ctr_test.csv)
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

AutoCross使用请参考文档 [AutoCross EMR](https://yuque.antfin-inc.com/pai/automl/cicak6)。

- AutoCross yaml配置文件：[ctr\_autocross.yaml](https://yuguang-test.oss-cn-beijing.aliyuncs.com/fe/configs/ctr_autocross.yaml)（[配置文件解析](https://yuque.antfin-inc.com/pai/automl/cicak6)）
- alink环境配置文件，另存为a[link.env](https://yuguang-test.oss-cn-beijing.aliyuncs.com/fe/configs/alink.env)

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

### 对接easy\_rec

Easy\_rec使用请参考文档 [EMR Tutorial](https://yuque.antfin.com/pai/arch/zucdp3)。
以下说明AutoCross后的数据对接easy\_rec的配置（[ctr\_deepmodel\_ac.config](https://yuguang-test.oss-cn-beijing.aliyuncs.com/fe/configs/ctr_deepmodel_ac.config)）

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
feature_configs: {
  input_names: "hour"
  feature_type: IdFeature
  embedding_dim: 16
  hash_bucket_size: 50
}
feature_configs: {
  input_names: "c1"
  feature_type: IdFeature
  embedding_dim: 16
  hash_bucket_size: 10
}
...
# 以下新增加的交叉列
feature_configs: {
  input_names: "cross_1"
  feature_type: IdFeature
  embedding_dim: 16
  hash_bucket_size: 500
}
feature_configs: {
  input_names: "cross_2"
  feature_type: IdFeature
  embedding_dim: 16
  hash_bucket_size: 500
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

使用el\_submit提交训练即可，请参照 [EMR Tutorial](https://yuque.antfin.com/pai/arch/zucdp3)。
