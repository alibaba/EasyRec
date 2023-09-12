# combo_feature

combo_feature是多个字段（或表达式）的组合（即笛卡尔积），id feature可以看成是一种特殊的combo feature，即参与交叉字段只有一个的combo feature。一般来讲，参与交叉的各个字段来自不同的表（比如user特征和item特征进行交叉）。

配置：

```
{
   "feature_type" : "combo_feature",
   "feature_name" : "comb_u_age_item",
   "expression" : ["user:age_class", "item:item_id"]
}
```

## 例子

^\]表示多值分隔符，注意这是一个符号，其ASCII编码是"\\x1D"，而不是两个符号

| user:age_class的取值 | item:item_id的取值 | 输出的feature                                                                                                 |
| ----------------- | --------------- | ---------------------------------------------------------------------------------------------------------- |
| 123               | 45678           | comb_u_age_item_123_45678                                                                                  |
| abc, bcd          | 45678           | comb_u_age_item_abc_45678, comb_u_age_item_bcd_45678                                                       |
| abc, bcd          | 12345^\]45678   | comb_u_age_item_abc_12345, comb_u_age_item_abc_45678, comb_u_age_item_bcd_12345, comb_u_age_item_bcd_45678 |

输出的feature个数等于

```
|F1| * |F2| * ... * |Fn|
```

其中Fn指依赖的第n个字段的值的个数。
