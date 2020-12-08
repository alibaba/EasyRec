特征
====

在上一节介绍了输入数据包括MaxCompute表、csv文件、hdfs文件、OSS文件等，表或文件的一列对应一个特征。

在数据中可以有一个或者多个label字段，而特征比较丰富，支持的类型包括IdFeature，RawFeature，TagFeature，SequenceFeature,
ComboFeature。

各种特征共用字段
----------------

-  input\_names: 输入的字段，根据特征需要，可以设置1个或者多个
-  feature\_name: 特征名称，如果没有设置，默认使用input\_names[0] 。
-  shared\_names:
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

-  其中embedding\_dim 的计算方法可以参考：

   .. math::


        embedding\_dim=8+x^{0.25}


-  hash\_bucket\_size: hash bucket的大小

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

连续值类特征可以先在pai-studio中先进行离散化，可以进行等频/等距/自动离散化，变成IdFeature。也可以将离散化的区间配置在config中，如下：

.. code:: protobuf

    feature_configs {
      input_names: "ctr"
      feature_type: RawFeature
      boundaries: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
      embedding_dim: 8
    }

-  boundaries: 分桶的值，通过一个数组来设置。
-  如果这个分割点来自pai-studio
   的分箱模型，需要根据代码读取分割点并设置值。参考：easy\_rec/python/tools/add\_boundaries\_to\_config.py
-  embedding\_dim: 如果设置了boundaries，则需要配置embedding dimension。
-  如果没有设置boundaries，在deepfm算法的wide端会被忽略

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

结合weights字段，可以描述用户的偏好类目和分数：

.. code:: protobuf

    feature_configs : {
       input_names: 'categories'
       input_names: 'scores'
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

ComboFeature：组合特征
----------------------

对输入的离散值进行组合, 如age + sex:

.. code:: protobuf

    feature_configs {
        input_names: ["age", "sex"]
        feature_type: ComboFeature
        embedding_dim: 16
        hash_bucket_size: 1000
    }

-  input\_names: 需要组合的特征名，数量>=2,
   来自data\_config.input\_fields.input\_name
-  embedding\_dim: embedding的维度，同IdFeature
-  hash\_bucket\_size: hash bucket的大小

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
