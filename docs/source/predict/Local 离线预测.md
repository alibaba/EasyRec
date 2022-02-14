# Local Prediction

## 方式一(推荐)

### 前置条件

- 模型训练
- 模型导出

### 离线预测

```bash
CUDA_VISIBLE_DEVICES=0 python -m easy_rec.python.predict --input_path 'data/test/tb_data/taobao_test_data' --output_path 'data/test/taobao_test_data_pred_result' --saved_model_dir experiments/dssm_taobao_ckpt/export/final --reserved_cols 'ALL_COLUMNS' --output_cols 'ALL_COLUMNS'
```

- save_modeld_dir: 导出的模型目录
- input_path: 输入文件路径
- output_path: 输出文件路径，不需要提前创建，会自动创建
- batch_size: minibatch的大小
- reserved_cols: 需要copy到输出文件的columns, 默认为'ALL_COLUMNS'，则所有的column都被copy到输出文件中
- output_cols: output_name和类型, 如:
  - 默认为'ALL_COLUMNS'
  - ctr模型(num_class=1)，导出字段:logits、probs，对应: sigmoid之前的值/概率，eg: output_cols="probs double"
  - 回归模型，导出字段: y，对应: 预测值，eg: output_cols="y double"
  - 多分类模型，导出字段: logits/probs/y，对应: softmax之前的值/概率/类别id
  - 如果有多列，用逗号分割, eg: output_cols='probs double,embedding string'
- input_sep: 输入文件的分隔符，默认","
- output_sep: 输出文件的分隔符，默认"\\u0001"

### 输出表schema

- 包含output_cols和reserved_cols

## 方式二

### 前置条件

- 模型训练
- 模型导出

### 加载模型

```
from easy_rec.python.inference.predictor import Predictor

# export is the directory of saved models
predictor = Predictor('model/export/')
```

输入格式:

1. list 格式
1. dict 格式
   输出是list of dict，dict里面包含一个字段y，即score: 范围在\[0, 1\]之间

```
# interface 1, input is a list of fields, the order of fields
# must be the same as that of data_config.input_fields
with open(test_path, 'r') as fin:
  reader = csv.reader(fin)
  inputs = []
  # the first is label, skip first column
  for row in reader:
    inputs.append(row[1:])
  output_res = predictor.predict(inputs, batch_size=32)
  assert len(output_res) == 63
  assert abs(output_res[0]['y'] - 0.5726) < 1e-3

# interface 2, input is a dict of fields
# the field_keys must be the same as data_config.input_fields.input_name
field_keys = [ "field1", "field2", "field3", "field4", "field5",
               "field6", "field7", "field8", "field9", "field10",
               "field11", "field12", "field13", "field14", "field15",
               "field16", "field17", "field18", "field19", "field20" ]
with open(test_path, 'r') as fin:
  reader = csv.reader(fin)
  inputs = []
  for row in reader:
    inputs.append({ f : row[fid+1] for fid, f in enumerate(field_keys) })
  output_res = predictor.predict(inputs, batch_size=32)
  assert len(output_res) == 63
  assert abs(output_res[0]['y'] - 0.5726) < 1e-3

```
