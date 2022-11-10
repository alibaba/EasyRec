# 输入输出

## 命令

```bash
   saved_model_cli show --all --dir export/1650854967
```

## 输出

```
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['adgroup_id'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_3:0
    inputs['age_level'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_12:0
    inputs['brand'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_7:0
    inputs['campaign_id'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_5:0
    inputs['cate_id'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_4:0
    inputs['cms_group_id'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_10:0
    inputs['cms_segid'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_9:0
    inputs['customer'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_6:0
    inputs['final_gender_code'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_11:0
    inputs['new_user_class_level'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_16:0
    inputs['occupation'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_15:0
    inputs['pid'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_2:0
    inputs['price'] tensor_info:
        dtype: DT_INT32
        shape: (-1)
        name: input_19:0
    inputs['pvalue_level'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_13:0
    inputs['shopping_level'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_14:0
    inputs['tag_brand_list'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_18:0
    inputs['tag_category_list'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_17:0
    inputs['user_id'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_8:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['item_emb'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: ReduceJoin_1/ReduceJoin:0
    outputs['item_tower_feature'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: ReduceJoin_3/ReduceJoin:0
    outputs['logits'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: Reshape:0
    outputs['probs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: Sigmoid:0
    outputs['user_emb'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: ReduceJoin/ReduceJoin:0
    outputs['user_tower_feature'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: ReduceJoin_2/ReduceJoin:0
  Method name is: tensorflow/serving/predict

```

- signature_def: 默认是serving_default
- inputs: 输入列表
  - dtype: 输入tensor类型
  - shape: 输入tensor的shape
  - name: 输入Placeholder的名称
- outputs: 输出列表
  - dtype: 输出tensor类型
  - shape: 输入tensor的shape
  - name: 输出tensor的名称
