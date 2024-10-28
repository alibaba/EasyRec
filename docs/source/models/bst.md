# BST

### 简介

利用近年因 Transformer 而备受关注的 Multi-head Self-attention，捕捉用户行为序列的序列信息。支持多组序列共同embedding，如hist_item_id, hist_category_id。

EasyRec提供两种使用`BST`模型的方法：

#### 1. 内置模型

内置模型目前结合`multi-tower`共同使用，bst 部分作为`multi-tower`的一个塔。

#### 2. 组件化模型（推荐）

使用组件化方法搭建标准的`BST`模型会更加方便，模型结构如下：

```
output:
                                	             probability of a click
model:
			                            	       /|\
				                          ______|______
		                             		 |             |
                            			      	 |             |
	                            	      		 |     MLP     |
                            			      	 |             |
	                            		      	 |_____________|
			                                        |
                              _____________________________>  ConCat  <______________________
                             |		        |                  	 |	            |
                          ___|__________________| _______________________|__                |
		   	|	                                            |	            |
		   	|	          Transformer Layer	            |	            |
		   	|___________________________________________________|	            |
		        	|	          |	             |	                    |
		        	|                 |	             |	                    |
                        |_Emb_|____|__|   |_Emb_|____|__| …… |_Emb_|____|__|        |_Emb_|____|__|
input:
		          target item	      item 1 	         item N	             other features
		          target item	      item 1 	         item N	               other features
                                           \______________________________/
                                                          |
                                                 User Behavior Sequence
```

### 模型配置

#### 1. 内置模型

```protobuf
model_config:{
  model_class: "MultiTowerBST"
  feature_groups: {
    group_name: 'user'
    feature_names: 'user_id'
    feature_names: 'cms_segid'
    feature_names: 'cms_group_id'
    feature_names: 'age_level'
    feature_names: 'pvalue_level'
    feature_names: 'shopping_level'
    feature_names: 'occupation'
    feature_names: 'new_user_class_level'
    wide_deep: DEEP
  }
  feature_groups: {
    group_name: 'item'
    feature_names: 'adgroup_id'
    feature_names: 'cate_id'
    feature_names: 'campaign_id'
    feature_names: 'customer'
    feature_names: 'brand'
    feature_names: 'price'
    feature_names: 'pid'
    wide_deep: DEEP
  }
  seq_att_groups: {
    group_name: "bst"
    seq_att_map: {
       key: "brand"
       hist_seq: "tag_brand_list"
    }
    seq_att_map: {
       key: "cate_id"
       hist_seq: "tag_category_list"
    }
  }
  multi_tower {
    towers {
      input: "user"
      dnn {
        hidden_units: [256, 128, 96, 64]
      }
    }
    towers {
      input: "item"
      dnn {
        hidden_units: [256, 128, 96, 64]
      }
    }
    bst_towers {
      input: "bst"
      seq_len: 50
      multi_head_size: 4
    }
    final_dnn {
      hidden_units: [128, 96, 64, 32, 16]
    }
    l2_regularization: 5e-7
  }
  embedding_regularization: 5e-5
}
```

- model_class: 'MultiTowerBST', 不需要修改。
- feature_groups: 可配置多个feature_group，group name可以变。
- seq_att_groups: 可配置多个seq_att_groups。
  - group name
  - seq_att_map: 需配置key和hist_seq，一一对应。
- multi_tower: multi_tower相关的参数。
  - towers: 每个feature_group对应了一个tower。
    - input必须和feature_groups的group_name对应
    - dnn: deep part的参数配置
      - hidden_units: dnn每一层的channel数目，即神经元的数目
  - bst_towers: 每个seq_att_groups对应了一个bst_tower。
    - input必须和seq_att_groups的group_name对应
    - seq_len: 历史序列的最大长度
    - multi_head_size: Multi-head Self-attention 中的 head size
  - final_dnn 整合towers和din_towers的输入。
    - hidden_units: dnn每一层的channel数目，即神经元的数目
- embedding_regularization: 对embedding部分加regularization，防止overfit

#### 2. 组件化模型

```protobuf
model_config: {
  model_name: 'BST'
  model_class: 'RankModel'
  feature_groups: {
    group_name: 'normal'
    feature_names: 'user_id'
    feature_names: 'cms_segid'
    feature_names: 'cms_group_id'
    feature_names: 'age_level'
    feature_names: 'pvalue_level'
    feature_names: 'shopping_level'
    feature_names: 'occupation'
    feature_names: 'new_user_class_level'
    feature_names: 'adgroup_id'
    feature_names: 'cate_id'
    feature_names: 'campaign_id'
    feature_names: 'customer'
    feature_names: 'brand'
    feature_names: 'price'
    feature_names: 'pid'
    wide_deep: DEEP
  }
  feature_groups: {
    group_name: 'sequence'
    feature_names: "cate_id"
    feature_names: "brand"
    feature_names: "tag_category_list"
    feature_names: "tag_brand_list"
    wide_deep: DEEP
  }
  backbone {
    blocks {
      name: 'deep'
      inputs {
        feature_group_name: 'normal'
      }
      keras_layer {
        class_name: 'MLP'
        mlp {
          hidden_units: [256, 128]
        }
      }
    }
    blocks {
      name: 'seq_input'
      inputs {
        feature_group_name: 'sequence'
      }
      input_layer {
        output_seq_and_normal_feature: true
      }
    }
    blocks {
      name: 'BST'
      inputs {
        block_name: 'seq_input'
      }
      keras_layer {
        class_name: 'BST'
        bst {
          hidden_size: 128
          num_attention_heads: 2
          num_hidden_layers: 2
          intermediate_size: 128
          hidden_act: 'gelu'
          max_position_embeddings: 50
          hidden_dropout_prob: 0.1
          attention_probs_dropout_prob: 0
        }
      }
    }
    top_mlp {
      hidden_units: [256, 128, 64]
    }
  }
  model_params {
    l2_regularization: 0
  }
  embedding_regularization: 0
}
```

- model_name: 任意自定义字符串，仅有注释作用
- model_class: 'RankModel', 不需要修改, 通过组件化方式搭建的单目标排序模型都叫这个名字
- feature_groups: 特征组
  - 包含两个feature_group: dense 和sparse group
  - wide_deep: BST模型使用的都是Deep features, 所以都设置成DEEP
  - 序列组件对应的feature_group的配置方式请查看 [参考文档](../component/sequence.md)
- backbone: 通过组件化的方式搭建的主干网络，[参考文档](../component/backbone.md)
  - blocks: 由多个`组件块`组成的一个有向无环图（DAG），框架负责按照DAG的拓扑排序执行个`组件块`关联的代码逻辑，构建TF Graph的一个子图
  - name/inputs: 每个`block`有一个唯一的名字（name），并且有一个或多个输入(inputs)和输出
  - input_layer: 对输入的`feature group`配置的特征做一些额外的加工，比如执行可选的`batch normalization`、`layer normalization`、`feature dropout`等操作，并且可以指定输出的tensor的格式（2d、3d、list等）；[参考文档](../component/backbone.md#id15)
  - keras_layer: 加载由`class_name`指定的自定义或系统内置的keras layer，执行一段代码逻辑；[参考文档](../component/backbone.md#keraslayer)
  - concat_blocks: DAG的输出节点由`concat_blocks`配置项定义，不配置时默认为所有DAG的叶子节点
  - top_mlp: 各输出`组件块`的输出tensor拼接之后输入给一个可选的顶部MLP层
- model_params:
  - l2_regularization: 对DNN参数的regularization, 减少overfit
- embedding_regularization: 对embedding部分加regularization, 减少overfit

### 示例config

1. 内置模型: [BST_demo.config](https://easyrec.oss-cn-beijing.aliyuncs.com/config/bst.config)
1. 组件化模型: [BST_backbone.config](https://github.com/alibaba/EasyRec/blob/master/samples/model_config/bst_backbone_on_taobao.config)

### 参考论文

[Behavior Sequence Transformer](https://arxiv.org/abs/1905.06874v1)
