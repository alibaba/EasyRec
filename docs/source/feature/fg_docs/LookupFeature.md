# 6.5 Lookup Feature

## 功能简介

如果离线生成不符合预期 请先使用最新的离线fg包

lookup feature 和 match feature类似，是从一组kv中匹配到自己需要的结果。

lookup feature 依赖 map 和 key 两个字段，map是一个多值string(MultiString)类型的字段，其中每一个string的样子如"k1:v2"。；key可以是一个任意类型的字段。生成特征时，先是取出key的值，将其转换成string类型，然后在map字段所持有的kv对中进行匹配，获取最终的特征。

map 和 key 源可以是 item，user，context 的任意组合。在线输入的时候item的多值用多值分隔符char(29)分隔，user和context的多值在tpp访问时用list表示。该特征仅支持json形式的配置方式。

## 实例

```json
{
    "features" : [
        {
            "feature_type" : "lookup_feature",
            "feature_name" : "item_match_item",
            "map" : "item:item_attr",
            "key" : "item:item_value",
            "needDiscrete" : true
        }
    ]
}
```

对于上面的配置，假设对于某个 doc：

```
item_attr : "k1:v1^]k2:v2^]k3:v3"
```

^\]表示多值分隔符，注意这是一个符号，其ASCII编码是"\\x1D"，而不是两个符号。该字符在emacs中的输入方式是C-q C-5, 在vi中的输入方式是C-v C-5。 这里item_attr是个多值string。需要切记，当map用来表征多个kv对时，是个多值string，而不是string！

```
item_value : "k2"
```

特征结果为 item_match_item_k2_v2。由于needDiscrete的值为true，所以特征结果为离散化后的结果。

## 其它

match feature 和 lookup feature都是匹配类型的特征，即从kv对中匹配到相应的结果。两者的区别是： match feature的被匹配字段user 必须是qinfo中传入的字段，即一次查询中对所有的doc来说这个字段的值都是一致的。而 lookup feature 的 key 和 map 没有来源的限制。

## 配置详解

默认情况的配置为 `needDiscrete == true, needWeighting = false, needKey = true, combiner = "sum"`

### 默认输出

### needWeighting == true

```
feature_name:fg
map:{{"k1:123", "k2:234", "k3:3"}}
key:{"k1"}
结果：feature={"fg_k1", 123}
```

此时会用 string 部分查 weight 表，然后乘对应 feature value 用于 LR 模型。

### needDiscrete == true

```
feature_name:fg
map:{{"k1:123", "k2:234", "k3:3"}}
key:{"k1"}
结果：feature={"fg_123"}
```

### needDiscrete == false

```
map:{{"k1:123", "k2:234", "k3:3"}}
key:{"k1"}
结果：feature={123}
```

如果存在多个 key 时，可以通过配置 combiner 来组合多个查到的值。可能的配置有 `sum, mean, max, min`。 ps：如果要使用combiner的话需要将needDiscrete设置为false，只有dense类才能做conbiner，生成的value会是数值类的

一个配置样例 update on 2021.04.15

```json
"kv_fields_encode": [
    {
      "name": "cnty_dense_features",
      "dimension": 99,
      "min_hash_type": 0,
      "use_sparse": true
    },
    {
      "name": "cross_a_tag",
      "dimension": 12,
      "min_hash_type": 0,
      "use_sparse": true
    },
    {
      "name": "cross_gender",
      "dimension": 12,
      "min_hash_type": 0,
      "use_sparse": true
    },
    {
      "name": "cross_purchasing_power",
      "dimension": 12,
      "min_hash_type": 0,
      "use_sparse": true
    }
  ]
```
