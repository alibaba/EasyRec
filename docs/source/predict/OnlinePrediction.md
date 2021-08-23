# Online Prediction

使用easy_rec提供的Predictor进行预测

### 加载模型

```bash
from easy_rec.python.inference.predictor import Predictor

# export is the directory of saved models
predictor = Predictor('model/export/')
```

### 输入格式

1. list 格式
1. dict 格式

输出是list of dict，dict里面包含一个字段y，即score: 范围在\[0, 1\]之间

```bash
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

### 使用EAS加载模型

可以使用TF Processor或者自定义Processor, 具体参考:[EAS加载模型](https://help.aliyun.com/document_detail/113696.html?spm=a2c4g.11186623.6.716.69da371b9G94HF)
