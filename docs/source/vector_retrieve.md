# 向量近邻检索

## Pai 命令

```sql
pai -name easy_rec_ext -project algo_public_dev
-Dcmd=vector_retrieve
-Dquery_table=odps://pai_online_project/tables/query_vector_table
-Ddoc_table=odps://pai_online_project/tables/doc_vector_table
-Doutput_table=odps://pai_online_project/tables/result_vector_table
-Dcluster='{"worker" : {"count":3, "cpu":600, "gpu":100, "memory":10000}}'
-Dknn_distance=inner_product
-Dknn_num_neighbours=100
-Dknn_feature_dims=128
-Dknn_index_type=gpu_ivfflat
-Dknn_feature_delimiter=','
-Dbuckets='oss://${oss_bucket}/'
-Darn='acs:ram::${xxxxxxxxxxxxx}:role/AliyunODPSPAIDefaultRole'
-DossHost='oss-cn-hangzhou-internal.aliyuncs.com'
```

## 参数说明

| 参数名                | 默认值        | 参数说明                                                                           |
| --------------------- | ------------- | ---------------------------------------------------------------------------------- |
| query_table           | 无            | 输入查询表, schema: (id bigint, vector string)                                     |
| doc_table             | 无            | 输入索引表, schema: (id bigint, vector string)                                     |
| output_table          | 无            | 输出表, schema: (query_id bigint, doc_id bigint, distance double)                  |
| knn_distance          | inner_product | 计算距离的方法：l2、inner_product                                                  |
| knn_num_neighbours    | 无            | top n, 每个query输出多少个近邻                                                     |
| knn_feature_dims      | 无            | 向量维度                                                                           |
| knn_feature_delimiter | ,             | 向量字符串分隔符                                                                   |
| knn_index_type        | ivfflat       | 向量索引类型：'flat', 'ivfflat', 'ivfpq', 'gpu_flat', 'gpu_ivfflat', 'gpu_ivfpg'   |
| knn_nlist             | 5             | 聚类的簇个数, number of split cluster on each worker                               |
| knn_nprobe            | 2             | 检索时只考虑距离与输入向量最近的簇个数, number of probe part on each worker        |
| knn_compress_dim      | 8             | 当index_type为`ivfpq` and `gpu_ivfpq`时, 指定压缩的维度，必须为float属性个数的因子 |

## 使用示例

### 1. 创建索引表

```sql
create table doc_table(pk BIGINT,vector string) partitioned by (pt string);

INSERT OVERWRITE TABLE doc_table PARTITION(pt='20190410')
VALUES
    (1, '0.1,0.2,-0.4,0.5'),
    (2, '-0.1,0.8,0.4,0.5'),
    (3, '0.59,0.2,0.4,0.15'),
    (10, '0.1,-0.2,0.4,-0.5'),
    (20, '-0.1,-0.2,0.4,0.5'),
    (30, '0.5,0.2,0.43,0.15')
;
```

### 2. 创建查询表

```sql
create table query_table(pk BIGINT,vector string) partitioned by (pt string);

INSERT OVERWRITE TABLE query_table PARTITION(pt='20190410')
VALUES
    (1, '0.1,0.2,0.4,0.5'),
    (2, '-0.1,0.2,0.4,0.5'),
    (3, '0.5,0.2,0.4,0.5'),
    (10, '0.1,0.2,0.4,0.5'),
    (20, '-0.1,-0.2,0.4,0.5'),
    (30, '0.5,0.2,0.43,0.15')
;
```

### 3. 执行向量检索

```sql
pai -name easy_rec_ext -project algo_public_dev
-Dcmd='vector_retrieve'
-Dquery_table='odps://${project}/tables/query_table/pt=20190410'
-Ddoc_table='odps://${project}/tables/doc_table/pt=20190410'
-Doutput_table='odps://${project}/tables/knn_result_table/pt=20190410'
-Dknn_distance=inner_product
-Dknn_num_neighbours=2
-Dknn_feature_dims=4
-Dknn_index_type='ivfflat'
-Dknn_feature_delimiter=','
-Dbuckets='oss://${oss_bucket}/'
-Darn='acs:ram::${xxxxxxxxxxxxx}:role/AliyunODPSPAIDefaultRole'
-DossHost='oss-cn-shenzhen-internal.aliyuncs.com'
-Dcluster='{
    \"worker\" : {
        \"count\" : 1,
        \"cpu\" : 600
    }
}';
```

FQA: 遇到以下错误怎么办？

```
File "run.py", line 517, in main
  raise ValueError('cmd should be one of train/evaluate/export/predict')
ValueError: cmd should be one of train/evaluate/export/predict
```

这个错误是因为包含`向量近邻检索`的最新的EasyRec版本暂时还没有正式发布。

解决方案：从 [Github](https://github.com/alibaba/EasyRec)
的master分支拉取最新代码，使用`bash pai_jobs/deploy_ext.sh -V ${version}`命令打一个最新的资源包`easy_rec_ext_${version}_res.tar.gz`，
上传到MaxCompute作为Archive资源，最后，在上述命令中加两个如下的参数即可解决。

```
-Dversion='${version}'
-Dres_project=${maxcompute_project}
```

### 4. 查看结果

```sql
SELECT * from knn_result_table where pt='20190410';

-- query	doc	distance
-- 1	3	0.17999999225139618
-- 1	1	0.13999998569488525
-- 2	2	0.5800000429153442
-- 2	1	0.5600000619888306
-- 3	3	0.5699999928474426
-- 3	30	0.5295000076293945
-- 10	30	0.10700000077486038
-- 10	20	-0.0599999874830246
-- 20	20	0.46000003814697266
-- 20	2	0.3800000250339508
-- 30	3	0.5370000004768372
-- 30	30	0.4973999857902527
```
