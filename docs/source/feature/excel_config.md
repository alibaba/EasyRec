# Excel特征配置

由于特征配置部分比较长，从头构建配置文件比较麻烦，我们提供从excel生成训练配置文件的方式。

### 命令

- excel模板:
  - [multi_tower](https://easyrec.oss-cn-beijing.aliyuncs.com/data/multi_tower_template.xls)
  - [deep_fm](https://easyrec.oss-cn-beijing.aliyuncs.com/data/deepfm_template.xls)
  - wide_and_deep同deepfm

```bash
python -m easy_rec.python.tools.create_config_from_excel --model_type multi_tower --excel_path multi_tower_template.xls --output_path multi_tower.config
```

- model_type

**NOTE**: --model_type必须要和excel模板匹配

- excel配置文件

--excel_path

- 输出config文件

--output_path

- 输入输出文件

  --train_input_path TRAIN_INPUT_PATH --eval_input_path EVAL_INPUT_PATH

- 默认数据文件(csv)列(column)分割符号是, 列(column)内部里面字符分割符号是|

  可以自定义分隔符:
  --column_separator $'|' --incol_separator $','

- 训练数据路径
  --train_input_path

- 评估数据路径
  --eval_input_path

- 模型目录
  --model_dir

### Excel配置说明

包含**features**, **global, group, types, basic_types** 5 个**sheet**

#### features:

特征配置

| **name** | **data_type** | **type** | **user_item_other** | **global** | **hash_bucket_size** | **embedding_dim** | **default_value** | **weights** | **boundaries** | **query** |
| -------- | ------------- | -------- | ------------------- | ---------- | -------------------- | ----------------- | ----------------- | ----------- | -------------- | --------- |
| label    | double        | label    | label               |            |                      |                   |                   |             |                |           |
| uid      | string        | category | user                | uid        |                      |                   |                   |             |                |           |
| own_room | bigint        | dense    | user                |            |                      |                   |                   |             | 10,20,30       |           |
| cate_idx | string        | tags     | user                | cate       |                      |                   |                   | cate_wgt    |                |           |
| cate_wgt | string        | weights  | user                |            |                      |                   |                   |             |                |           |
| **...**  |               |          |                     |            |                      |                   |                   |             |                |           |

- name: 输入列名
- data_type: 输入的数据类型，包含double, string, bigint, 应用[basic_types](#6QphS)
- type: 特征类型(引用[types](#78xRB) sheet types列)
  - label: 要预测的列
  - category: 离散值特征
  - dense: 连续值特征
  - tags: 标签型特征
    - 关键词默认使用|分割，如使用其他分割符, 可以通过--incol_separator指定
  - weights: tags对应的weight
  - indexes: 一串数字，使用incol_separator分割, 如: 1|2|4|5
  - notneed: 不需要的，可以忽略的
- group(引用group sheet列)
  - multi_tower
    - user: user tower
    - item: item tower
    - user_item: user_item tower
    - label: label tower
  - deepfm
    - wide: 特征仅用在wide部分
    - deep: 特征仅用在deep和fm部分
    - wide_and_deep: 特征用在wide, deep, fm部分，默认选wide_and_deep
- global(引用[global](#ap1R1)里面的name列)

global相同的列share embedding

- hash_bucket_size: hash_bucket桶的大小
- embedding_dim: embedding的大小
  - **NOTE**: 对于deepfm，所有特征的embedding_dim必须是一样大的
- default_value: 缺失值填充
- weights: 如果type是tags，则可以指定weights, weights和tags必须要有相同的列
- boundaries: 连续值离散化的区间，如: 10,20,30，将会离散成区间(-inf, 10), \[10, 20), \[20, 30), \[30, inf)
  - **NOTE**: 配置了boundaries，一般也要配置embedding_dim
  - **NOTE**: 对于deepfm，连续值必须要配置boundaries
- query: 当前未使用，拟用作DIN的target，通常是item_id
- **NOTE**:
  - features必须和odps表或者csv文件的列是**一一对应的**，**顺序必须要一致**
  - features的数目和输入标或者文件的列的数目必须是一致的
  - 如果某些列不需要的话，可以设置type为notneed

#### types

描述数据类型

| **types** |
| --------- |
| label     |
| category  |
| tags      |
| weights   |
| dense     |
| indexes   |
| notneed   |

#### basic_types

描述输入表里面的数据类型， 包含string, bigint, double

| **basic_types** |
| --------------- |
| string          |
| bigint          |
| double          |

#### global

描述share embedding里面share embedding的信息
其中hash_bucket_size embedding_dim default_value会覆盖features表里面对应的信息

| **name** | **type** | **hash_bucket_size** | **embedding_dim** | **default_value** |
| -------- | -------- | -------------------- | ----------------- | ----------------- |
| cate     | category | 1000                 | 10                | 0                 |
| uid      | category | 100000               | 10                | 0                 |
| **...**  |          |                      |                   |                   |

#### group

- deepfm模型中的分组设置

  | **group**     |
  | ------------- |
  | wide_and_deep |
  | wide          |
  | deep          |
  | label         |

- multi_tower模型中的分组设置

  | **group** |
  | --------- |
  | user      |
  | item      |
  | user_item |
  | label     |

  - 其中user, item, user_item可以自定义

- 其它模型的分组设置暂不支持
