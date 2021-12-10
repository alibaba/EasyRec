特征
====

在上一节介绍了输入数据包括MaxCompute表、csv文件、hdfs文件、OSS文件等，表或文件的一列对应一个特征。

在数据中可以有一个或者多个label字段，而特征比较丰富，支持的类型包括IdFeature，RawFeature，TagFeature，SequenceFeature,
ComboFeature。

各种特征共用字段
----------------

-  **input\_names**: 输入的字段，根据特征需要，可以设置1个或者多个
-  **feature\_name**: 特征名称，如果没有设置，默认使用input\_names[0]作为feature\_name
  - 如果有多个特征使用同一个input\_name，则需要设置不同的feature\_name, 否则会导致命名冲突
  .. code:: protobuf

     feature_configs {
       input_names: "uid"
       feature_type: IdFeature
       embedding_dim: 32
       hash_bucket_size: 100000
     }

     feature_configs {
       feature_name: "combo_uid_category"
       input_names: "uid"
       input_names: "category"
       feature_type: ComboFeature
       embedding_dim: 32
       hash_bucket_size: 1000000
     }

-  **shared\_names**:
   其它输入的数据列，复用这个config，仅仅适用于只有一个input\_names的特征，不适用于有多个input\_names的特征，如ComboFeature。

IdFeature: 离散值特征/ID类特征
------------------------------

离散型特征，例如手机品牌、item\_id、user\_id、年龄段、星座等，一般在表里面存储的类型一般是string或者bigint。

.. code:: protobuf

    feature_configs {
      input_names: "uid"
      feature_type: IdFeature
      embedding_dim: 32
      hash_bucket_size: 100000
    }

    feature_configs {
      input_names: "month"
      feature_type: IdFeature
      embedding_dim: 8
      num_buckets: 12
    }

    feature_configs {
      input_names: "weekday"
      feature_type: IdFeature
      embedding_dim: 8
      vocab_list: ["1", "2", "3", "4", "5", "6", "7"]
    }

-  其中embedding\_dim 的计算方法可以参考：

   .. math::


        embedding\_dim=8+x^{0.25}


-  hash\_bucket\_size: hash bucket的大小。适用于category_id, user_id等

-  对于user\_id等规模比较大的，hash冲突影响比较小的特征，

   .. math::


          hash\_bucket\_size  = \frac{number\_user\_ids}{ratio},      建议：ratio \in [10,100];


-  对于星座等规模比较小的，hash冲突影响比较大的

   .. math::


          hash\_bucket\_size = number\_xingzuo\_ids * ratio,    建议 ratio \in [5,10]


-  num\_buckets: buckets number,
   仅仅当输入是integer类型时，可以使用num\_buckets

-  vocab\_list:
   指定词表，适合取值比较少可以枚举的特征，如星期，月份，星座等

-  vocab\_file:
   使用文件指定词表，用于指定比较大的词表。在提交tf任务到pai集群的时候，可以把词典文件存储在oss中。

-  NOTE: hash\_bucket\_size, num\_buckets, vocab\_list,
   vocab\_file只能指定其中之一，不能同时指定

RawFeature：连续值特征
----------------------

连续值类特征可以先使用分箱组件+进行离散化，可以进行等频/等距/自动离散化，变成离散值。推荐使用分箱组件得到分箱信息表，在训练时可以通过"-Dboundary\_table odps://project_name/tables/boundary\_info"导入boundary\_info表，省去在config中写入boundaries的操作。

.. code:: protobuf

   DROP table if exists boundary_info;
   PAI -name binning
   -project algo_public
   -DinputTableName=train_data
   -DoutputTableName=boundary_info
   -DselectedColNames=col1,col2,col3,col4,col5
   -DnDivide=20;

   pai -name easy_rec_ext -project algo_public
    -Dconfig=oss://easyrec/config/MultiTower/dwd_avazu_ctr_deepmodel_ext.config
    -Dcmd=train
    -Dtables=odps://pai_online_project/tables/dwd_avazu_ctr_deepmodel_train,odps://pai_online_project/tables/dwd_avazu_ctr_deepmodel_test
    -Dboundary_table=odps://pai_online_project/tables/boundary_info
    -Dcluster='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":3, "cpu":1000, "gpu":100, "memory":40000}}'
    -Darn=acs:ram::xxx:role/xxx
    -Dbuckets=oss://easyrec/
    -DossHost=oss-cn-beijing-internal.aliyuncs.com
    -Dwith_evaluator=1;

.. code:: protobuf

    feature_configs {
      input_names: "ctr"
      feature_type: RawFeature
      embedding_dim: 8
    }

分箱组件使用方法见： `机器学习组件 <https://help.aliyun.com/document_detail/54352.html>`_
也可以手动导入分箱信息。如下：

.. code:: protobuf

    feature_configs {
      input_names: "ctr"
      feature_type: RawFeature
      boundaries: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
      embedding_dim: 8
    }

-  boundaries: 分桶的值，通过一个数组来设置。如果通过"-Dboundary\_table"导入分箱表，则无需写入，程序会自动导入到pipeline.config中。
-  embedding\_dim: 如果设置了boundaries，则需要配置embedding dimension。
-  如果没有设置boundaries，在deepfm算法的wide端会被忽略


这里同样支持embedding特征，如"0.233\|0.123\|0.023\|2.123\|0.233\|0.123\|0.023\|2.123"

.. code:: protobuf

    feature_configs {
      input_names: "pic_emb"
      feature_type: RawFeature
      separator: '|'
      raw_input_dim: 8
    }

- raw_input_dim: 指定embedding特征的维度

TagFeature
----------

标签类型特征,
在表里面存储的类型一般是string类型。格式一般为“XX\|XX\|XX”，如文章标签特征为“娱乐\|搞笑\|热门”，其中\|为分隔符。

有多个tag的情况下，tag之前使用分隔符进行分隔。

tags字段可以用于描述商品的多个属性

.. code:: protobuf

    feature_configs : {
       input_names: 'properties'
       feature_type: TagFeature
       separator: '|'
       hash_bucket_size: 100000
       embedding_dim: 24
    }

-  separator: 分割符，默认为'\|'
-  hash\_bucket\_size: hash分桶大小，配置策略和IdFeature类似
-  num\_buckets: 针对输入是整数的情况,
   如6\|20\|32，可以配置num\_buckets，配置为最大值
-  embedding\_dim: embedding的dimension，和IdFeature类似

我们同样支持有权重的tag特征，如"体育:0.3\|娱乐:0.2\|军事:0.5"：

.. code:: protobuf

    feature_configs : {
       input_names: 'tag_kvs'
       feature_type: TagFeature
       separator: '|'
       kv_separator: ':'
       hash_bucket_size: 100000
       embedding_dim: 24
    }
或"体育\|娱乐\|军事"和"0.3\|0.2\|0.5"的输入形式：

.. code:: protobuf

    feature_configs : {
       input_names: 'tags'
       input_names: 'tag_scores'
       feature_type: TagFeature
       separator: '|'
       hash_bucket_size: 100000
       embedding_dim: 24
    }

NOTE:
~~~~~

-  如果使用csv文件进行存储，那么多个tag之间采用\ **列内分隔符**\ 进行分隔，
   例如：csv的列之间一般用逗号(,)分隔，那么可采用竖线(\|)作为多个tag之间的分隔符。
-  weights：tags对应的权重列，在表里面一般采用string类型存储。
-  Weights的数目必须要和tag的数目一致，并且使用\ **列内分隔符**\ 进行分隔。

SequenceFeature：行为序列类特征
-------------------------------

Sequense类特征格式一般为“XX\|XX\|XX”，如用户行为序列特征为"item\_id1\|item\_id2\|item\_id3",
其中\|为分隔符，如:

.. code:: protobuf

    feature_configs {
      input_names: "play_sequence"
      feature_type: SequenceFeature
      embedding_dim: 32
      hash_bucket_size: 100000
    }

-  embedding\_dim: embedding的dimension
-  hash\_bucket\_size: 同离散值特征
-  NOTE：SequenceFeature一般用在DIN算法或者BST算法里面。

在模型中可支持对序列特征使用TextCNN算子进行embedding聚合，示例如下：

.. code:: protobuf
  feature_configs: {
    input_names: 'title'
    feature_type: SequenceFeature
    separator: ' '
    embedding_dim: 32
    hash_bucket_size: 10000
    sequence_combiner: {
      text_cnn: {
        filter_sizes: [2, 3, 4]
        num_filters: [16, 8, 8]
      }
    }
  }

- num_filters: 卷积核个数列表
- filter_sizes: 卷积核步长列表

TextCNN网络是2014年提出的用来做文本分类的卷积神经网络，由于其结构简单、效果好，在文本分类、推荐等NLP领域应用广泛。
从直观上理解，TextCNN通过一维卷积来获取句子中`N gram`的特征表示。
在推荐模型中，可以用TextCNN网络来提取文本类型的特征。

ComboFeature：组合特征
----------------------

对输入的离散值进行组合, 如age + sex:

.. code:: protobuf

    feature_configs {
        input_names: ["age", "sex"]
        feature_name: "combo_age_sex"
        feature_type: ComboFeature
        embedding_dim: 16
        hash_bucket_size: 1000
    }

-  input\_names: 需要组合的特征名，数量>=2,
   来自data\_config.input\_fields.input\_name
-  embedding\_dim: embedding的维度，同IdFeature
-  hash\_bucket\_size: hash bucket的大小

特征选择
------
对输入层使用变分dropout计算特征重要性，根据重要性排名进行特征选择。

rank模型中配置相应字段：

.. code:: protobuf

    variational_dropout{
        regularization_lambda:0.01
        embedding_wise_variational_dropout:false
    }

-  regularization\_lambda: 变分dropout层的正则化系数设置
-  embedding\_wise\_variational\_dropout: 变分dropout层维度是否为embedding维度（true：embedding维度；false：feature维度；默认false）






分隔符
------

列间分隔符
~~~~~~~~~~

-  csv文件默认采用半角逗号作为分隔符
-  可以自定义分隔符，对应需要修改data\_config的separator字段

列内分隔符
~~~~~~~~~~

-  TagFeature和SequenceFeature特征需要用到列内分隔符，默认是\|
-  可以自定义，对应需要修改feature\_config的separator字段
