# overlap_feature

## 功能介绍

用来输出一些字符串字词匹配信息的feature

离线推荐使用1.3.56-SNAPSHOT这个版本,或者1.3.28（不支持参数need_prefix） ps: 写fg的时候注意维度，title的维度要大于或等于query的问题（简单来说就是如果title是user特征，那query也只能是user特征，user特征的batch size为1，商品特征的batch size为商品数）

| 方式                  | 描述                                              | 备注                 |
| ------------------- | ----------------------------------------------- | ------------------ |
| common_word         | 计算query与title间重复term，并输出为fg_common1_common2     | 重复数不超过query term数  |
| diff_word           | 计算query与title间不重复term，并输出为fg_diff1_diff2        | 不重复数不超过query term数 |
| query_common_ratio  | 计算query与title间重复term数占query中term比例,乘以10取下整      | 取值为\[0,10\]        |
| title_common_ratio  | 计算query与title间重复term数占title中term比例,乘以100取下整     | 取值为\[0,100\]       |
| is_contain          | 计算query是否全部包含在title中，保持顺序                       | 0表示未包含，1表示包含       |
| is_equal            | 计算query是否与title完全相同                             | 0表示不完全相同，1表示完全相同   |
| common_word_divided | 计算query与title间重复term，并输出为fg_common1, fg_common2 | 重复数不超过query term数  |
| diff_word_divided   | 计算query与title间不重复term，并输出为fg_diff1, fg_diff2    | 重复数不超过query term数  |

## 配置方法

```json
  {
			"feature_type" : "overlap_feature",
			"feature_name" : "is_contain",
			"query" : "user:attr1",
			"title" : "item:attr2",
			"method" : "is_contain",
			"separator" : " "
  }
```

| 字段名          | 含义                                                                                     |
| ------------ | -------------------------------------------------------------------------------------- |
| feature_type | 必选项，描述改feature的类型                                                                      |
| feature_name | 必选项，feature_name会被当做最终输出的feature的前缀                                                    |
| query        | 必选项，query依赖的表, attr1是一个多值string, 多值string的分隔符使用chr(29)                                 |
| title        | 必选项，title依赖的表, attr2是一个多值string                                                        |
| method       | 可填common_word, diff_word, query_common_ratio, title_common_ratio, is_contain， 对应上图五种方式 |
| separator    | 输出结果中的分割字符，不填写我们默认为\_ ，但也可以用户自己定制，具体看例子                                                |

## 例子

query为high,high2,fiberglass,abc
title为high,quality,fiberglass,tube,for,golf,bag

| method              | separator | feature                    |
| ------------------- | --------- | -------------------------- |
| common_word         |           | name_high_fiberglass       |
| diff_word           | " "       | name high2 abc             |
| query_common_ratio  |           | name_5                     |
| title_common_ratio  |           | name_28                    |
| is_contain          |           | name_0                     |
| is_equal            |           | name_0                     |
| common_word_divided |           | name_high, name_fiberglass |
| diff_word_divided   |           | name_high2, name_abc       |
