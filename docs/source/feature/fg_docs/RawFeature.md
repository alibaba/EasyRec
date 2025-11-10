# raw_feature

## 功能介绍

raw_feature表示连续值特征, 支持数值int、float、double等数值类型. 支持单值连续特征和多值连续特征.

## 配置方法

```json
{
 "feature_type" : "raw_feature",
 "feature_name" : "ctr",
 "expression" : "item:ctr",
 "normalizer" : "method=log10"
}
```

| 字段名          | 含义                                                                                   |
| --------------- | -------------------------------------------------------------------------------------- |
| feature_name    | 必选项, 特征名                                                                         |
| expression      | 必选项，expression描述该feature所依赖的字段来源, 来源必须是user、item、context中的一种 |
| value_dimension | 可选项，默认值为1，表示输出的字段的维度                                                |
| normalizer      | 可选项，归一化方法，详见后文                                                           |

## 示例

^\]表示多值分隔符，注意这是一个符号，其ASCII编码是"\\x1D"，而不是两个符号

| 类型    | item:ctr的取值 | 输出的feature                                                  |
| ------- | -------------- | -------------------------------------------------------------- |
| int64_t | 100            | (ctr, 100)                                                     |
| double  | 100.1          | (ctr, 100.1)                                                   |
| 多值int | 123^\]456      | (ctr, (123,456)) (注意，输入字段必须与配置的dimension维度一致) |

## Normalizer

raw_feature 和 match_feature 支持 normalizer，共三种，`minmax，zscore，log10`. 配置和计算方法如下:

### log10

```
配置例子: method=log10,threshold=1e-10,default=-10
计算公式: x = x > threshold ? log10(x) : default;
```

### zscore

```
配置例子: method=zscore,mean=0.0,standard_deviation=10.0
计算公式: x = (x - mean) / standard_deviation
```

### minmax

```
配置例子: method=minmax,min=2.1,max=2.2
计算公式: x = (x - min) / (max - min)
```
