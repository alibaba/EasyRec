# 6.1 Id Feature

功能介绍

id feature是一个sparse feature，是一种最简单的离散特征，只是简单的将某个字段的值与用户配置的feature名字拼接。

配置方法

```json
{
  "feature_type" : "id_feature",
  "feature_name" : "item_is_main",
  "expression" : "item:is_main"
}
```

| 字段名            | 含义                                                                            |
| -------------- | ----------------------------------------------------------------------------- |
| feature_name   | 必选项，feature_name会被当做最终输出的feature的前缀                                           |
| expression     | 必选项，expression描述该feature所依赖的字段来源                                              |
| need_prefix    | 可选项，true表示会拼上feature_name作为前缀，false表示不拼，默认为true，通常在shared_embedding的场景会用false |
| invalid_values | 可选项，表示这些values都会被输出成null。list string，例如\[""\]，表示将所有的空字符串输出变成null。             |

例子 （  ^\]表示多值分隔符，注意这是一个符号，其ASCII编码是"\\x1D"，而不是两个符号）

| 类型       | item:is_main的取值 | 输出的feature                                  |
| -------- | --------------- | ------------------------------------------- |
| int64_t  | 100             | (item_is_main_100, 1)                       |
| double   | 5.2             | (item_is_main_5, 1)（小数部分会被截取）               |
| string   | abc             | (item_is_main_abc, 1)                       |
| 多值string | abc^\]bcd       | (item_is_main_abc, 1),(item_is_main_bcd, 1) |
| 多值int    | 123^\]456       | (item_is_main_123, 1),(item_is_main_456, 1) |
