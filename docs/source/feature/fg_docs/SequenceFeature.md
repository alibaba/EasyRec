# sequence_feature

## 功能介绍

⽤户的历史⾏为也是⼀个很重要的 feature。历史⾏为通常是⼀个序列，例如点击序列、购买序列等，组成这个序列的实体可能是商品本身。

## 配置方法

例如我们需要对⽤户的点击序列进⾏fg，序列⻓度为50，每个序列提取item_id, price和ts特征，其中ts=请求时间(request_time) - 用户行为时间(event_time)。 配置如下：

```json
{
    "features":[
        {
            "feature_type":"raw_feature",
            "feature_name":"feat0",
            "expression":"user:feat0"
        },
        ...
        {
            "sequence_name":"click_50_seq",
            "sequence_column":"click_50_seq",
            "sequence_length":10,
            "sequence_delim":";",
            "attribute_delim":"#",
            "sequence_table":"item",
            "sequence_pk":"user:click_50_seq",
            "features":[
                {
                    "feature_name":"item_id",
                    "feature_type":"id_feature",
                    "value_type":"String",
                    "expression":"item:item_id"
                },
                {
                    "feature_name":"price",
                    "feature_type":"raw_feature",
                    "expression":"item:price"
                },
                {
                    "feature_name":"ts",
                    "feature_type":"raw_feature",
                    "expression":"user:ts"
                }
            ]
        }
    ]
}
```

- sequence_name: sequence名称
- sequence_column: sequence输出名成
- sequence_length: sequence的最大长度
- sequence_delim: sequence元素之间的分隔符
- attribute_delim: sequence元素内部各个属性之间的分隔符, 仅离线需要
- sequence_pk: sequence primary key, 主键, 如user:click_50_seq, 里面保存了user点击的最近的50个itemId;
- features: sequence的sideinfo, 包含item的静态属性值和行为时间信息等

### 在线 FG

⽀持两种⽅式获取⾏为sideinfo信息，⼀种是从EasyRec Processor的item cache获取sideinfo信息, 以`sequence_pk` 配置的字段为主键，[EasyRec Processor](../../predict/processor.md) 从item cache中查找item的属性信息; 另⼀种⽤户在请求中填充对应的字段值, 如上述配置中的"ts"字段, 其含义是(request_time - event_time), 即推荐请求时间 - 用户行为时间, 这个是随请求时间变化的, 因此需要从请求中获取:

```protobuf
user_features {
  key: "click_50_seq"
  value {
    string_feature: "9008721;34926279;22487529;73379;840804;911247;31999202;7421440;4911004;40866551"
  }
}

user_features {
  key: "click__ts"
  value {
    string_feature: "23;113;401363;401369;401375;401405;486678;486803;486922;486969"
  }
}
```
