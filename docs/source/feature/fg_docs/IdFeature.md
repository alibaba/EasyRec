# id_feature

## 功能介绍

id_feature表示离散特征, 包含单值离散特征和多值离散特征.

## 配置方法

```json
{
  "feature_type" : "id_feature",
  "feature_name" : "item_is_main",
  "expression" : "item:is_main"
}
```

| 字段名          | 含义                                                                            |
| ------------ | ----------------------------------------------------------------------------- |
| feature_name | 必选项，feature_name会被当做最终输出的feature的前缀                                           |
| expression   | 必选项，expression描述该feature所依赖的字段来源                                              |
| need_prefix  | 可选项，true表示会拼上feature_name作为前缀，false表示不拼，默认为true，通常在shared_embedding的场景会用false |

## 示例:

下面以item侧的特征is_main作为案例来说明在不同配置下特征的输入和输出:

| 类型       | item:is_main的取值 | 输出的feature                         |
| -------- | --------------- | ---------------------------------- |
| int64_t  | 100             | item_is_main_100                   |
| double   | 5.2             | item_is_main_5（小数部分会被截取）           |
| string   | abc             | item_is_main_abc                   |
| 多值string | abc^\]bcd       | item_is_main_abc^Citem_is_main_bcd |
| 多值int    | 123^\]456       | item_is_main_123^Citem_is_main_456 |

- ^\]表示多值分隔符，注意这是一个符号，其ASCII编码是"\\x1D", 也可以写作"\\u001d"
- ^C是FG encode之后输出的特征值的分隔符, 其ASCII编码是"\\x03"
