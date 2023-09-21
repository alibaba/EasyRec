# 组件详细参数

## 1.基础组件

- MLP （多层感知机）

| 参数                      | 类型   | 默认值        | 说明                          |
| ----------------------- | ---- | ---------- | --------------------------- |
| hidden_units            | list |            | 各隐层单元数                      |
| dropout_ratio           | list |            | 各隐层dropout rate             |
| activation              | str  | relu       | 每层的激活函数                     |
| use_bn                  | bool | true       | 是否使用batch normalization     |
| use_final_bn            | bool | true       | 最后一层是否使用batch normalization |
| use_bias                | bool | false      | 是否使用偏置项                     |
| use_final_bias          | bool | false      | 最后一层是否使用偏置项                 |
| final_activation        | str  | relu       | 最后一层的激活函数                   |
| initializer             | str  | he_uniform | 权重初始化方法，参考keras Dense layer |
| use_bn_after_activation | bool | false      | 是否在激活函数之后做batch norm        |

- HighWay

| 参数             | 类型     | 默认值  | 说明           |
| -------------- | ------ | ---- | ------------ |
| emb_size       | uint32 | None | embedding维度  |
| activation     | str    | gelu | 激活函数         |
| dropout_rate   | float  | 0    | dropout rate |
| init_gate_bias | float  | -3.0 | 门控网络的bias初始值 |
| num_layers     | int    | 1    | 网络层数         |

- PeriodicEmbedding

| 参数                 | 类型     | 默认值   | 说明                                                |
| ------------------ | ------ | ----- | ------------------------------------------------- |
| embedding_dim      | uint32 |       | embedding维度                                       |
| sigma              | float  |       | 初始化自定义参数时的标准差，**效果敏感、小心调参**                       |
| add_linear_layer   | bool   | true  | 是否在embedding之后添加额外的层                              |
| linear_activation  | str    | relu  | 额外添加的层的激活函数                                       |
| output_tensor_list | bool   | false | 是否同时输出embedding列表                                 |
| output_3d_tensor   | bool   | false | 是否同时输出3d tensor, `output_tensor_list=true`时该参数不生效 |

- AutoDisEmbedding

| 参数                 | 类型     | 默认值   | 说明                                                |
| ------------------ | ------ | ----- | ------------------------------------------------- |
| embedding_dim      | uint32 |       | embedding维度                                       |
| num_bins           | uint32 |       | 虚拟分桶数量                                            |
| keep_prob          | float  | 0.8   | 残差链接的权重                                           |
| temperature        | float  |       | softmax函数的温度系数                                    |
| output_tensor_list | bool   | false | 是否同时输出embedding列表                                 |
| output_3d_tensor   | bool   | false | 是否同时输出3d tensor, `output_tensor_list=true`时该参数不生效 |

## 2.特征交叉组件

- Bilinear

| 参数               | 类型     | 默认值         | 说明         |
| ---------------- | ------ | ----------- | ---------- |
| type             | string | interaction | 双线性类型      |
| use_plus         | bool   | true        | 是否使用plus版本 |
| num_output_units | uint32 |             | 输出size     |

- FiBiNet

| 参数       | 类型       | 默认值 | 说明               |
| -------- | -------- | --- | ---------------- |
| bilinear | Bilinear |     | protobuf message |
| senet    | SENet    |     | protobuf message |
| mlp      | MLP      |     | protobuf message |

## 3.特征重要度学习组件

- SENet

| 参数                    | 类型     | 默认值  | 说明                 |
| --------------------- | ------ | ---- | ------------------ |
| reduction_ratio       | uint32 | 4    | 隐层单元数量缩减倍数         |
| num_squeeze_group     | uint32 | 2    | 压缩分组数量             |
| use_skip_connection   | bool   | true | 是否使用残差连接           |
| use_output_layer_norm | bool   | true | 是否在输出层使用layer norm |

- MaskBlock

| 参数               | 类型     | 默认值  | 说明                              |
| ---------------- | ------ | ---- | ------------------------------- |
| output_size      | uint32 |      | 输出层单元数                          |
| reduction_factor | float  |      | 隐层单元数缩减因子                       |
| aggregation_size | uint32 |      | 隐层单元数                           |
| input_layer_norm | bool   | true | 输入是否需要做layer norm               |
| projection_dim   | uint32 |      | 用两个小矩阵相乘代替原来的输入-隐层权重矩阵，配置小矩阵的维数 |

- MaskNet

| 参数           | 类型   | 默认值  | 说明            |
| ------------ | ---- | ---- | ------------- |
| mask_blocks  | list |      | MaskBlock结构列表 |
| use_parallel | bool | true | 是否使用并行模式      |
| mlp          | MLP  | 可选   | 顶部mlp         |

## 4. 序列特征编码组件

- SeqAugment

| 参数           | 类型   | 默认值  | 说明            |
| ------------ | ---- | ---- | ------------- |
| mask_rate  | float | 0.6 | 被mask掉的token比率 |
| crop_rate | float | 0.2 | 裁剪保留的token比率     |
| reorder_rate | float | 0.6 | shuffle的子序列长度占比         |

请参考Protobuf Message的定义，文件路径：`easy_rec/python/protos/seq_encoder.proto`

## 5. 多任务学习组件

- MMoE

| 参数         | 类型     | 默认值 | 说明           |
| ---------- | ------ | --- | ------------ |
| num_task   | uint32 |     | 任务数          |
| num_expert | uint32 | 0   | expert数量     |
| expert_mlp | MLP    | 可选  | expert的mlp参数 |

## 6. 计算辅助损失函数的组件

- AuxiliaryLoss

| 参数          | 类型     | 默认值 | 说明                          |
| ----------- | ------ | --- | --------------------------- |
| loss_type   | string |     | 损失函数类型，包括：l2_loss, nce_loss, info_nce |
| loss_weight | float  | 1.0 | 损失函数权重                      |
| temperature | float  | 0.1 | info_nce loss 的参数           |
| 其他          |        |     | 根据loss_type决定               |
