# 向量近邻检索

## Pai 命令

```sql
pai -name easy_rec_ext -project algo_public
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
```

## 参数说明

| 参数名                   | 默认值           | 参数说明                                                                      |
| --------------------- | ------------- | ------------------------------------------------------------------------- |
| query_table           | 无             | 输入查询表, schema: (id bigint, vector string)                                 |
| doc_table             | 无             | 输入索引表, schema: (id bigint, vector string)                                 |
| output_table          | 无             | 输出表, schema: (query_id bigint, doc_id bigint, distance double)            |
| knn_distance          | inner_product | 计算距离的方法：l2、inner_product                                                  |
| knn_num_neighbours    | 无             | top n, 每个query输出多少个近邻                                                     |
| knn_feature_dims      | 无             | 向量维度                                                                      |
| knn_feature_delimiter | ,             | 向量字符串分隔符                                                                  |
| knn_index_type        | ivfflat       | 向量索引类型：'flat', 'ivfflat', 'ivfpq', 'gpu_flat', 'gpu_ivfflat', 'gpu_ivfpg' |
| knn_nlist             | 5             | number of split part on each worker                                       |
| knn_nprobe            | 2             | number of probe part on each worker                                       |
| knn_compress_dim      | 8             | number of dimensions after compress for `ivfpq` and `gpu_ivfpq`           |
