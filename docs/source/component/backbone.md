# 为何需要组件化

## 1. 依靠动态可插拔的公共组件，方便为现有模型添加新特性。

过去一个新开发的公共可选模块，比如`Dense Feature Embedding Layer`、 `SENet`添加到现有模型中，需要修改所有模型的代码才能用上新的特性，过程繁琐易出错。随着模型数量和公共模块数量的增加，为所有模型集成所有公共可选模块将产生组合爆炸的不可控局面。组件化实现了底层公开模块与上层模型的解耦。

## 2. 通过重组已有组件，实现“搭积木”式新模型开发。

很多模型之所以被称之为一个新的模型，是因为引入了一个或多个特殊的子模块（组件），然而这些子模块并不仅仅只能用在该模型中，通过组合各个不同的子模块可以轻易组装一个新的模型。组件化EasyRec支持通过配置化的方式搭建新的模型。

## 3. 添加新的特性将变得更加容易。

现在我们只需要为新的特征开发一个Keras Layer类，并在指定package中添加import语句，框架就能自动识别并添加到组件库中，不需要额外操作。开发一个新的模型，只需要实现特殊的新模块，其余部分可以通过组件库中的已有组件拼装。新人不再需要熟悉EasyRec的方方面面就可以为框架添加功能，开发效率大大提高。

# 组件化的目标

> 不再需要实现新的模型，只需要实现新的组件！ 模型通过组装组件完成。

各个组件专注自身功能的实现，模块中代码高度聚合，只负责一项任务，也就是常说的单一职责原则。

# 主干网络

组件化EasyRec模型使用一个可配置的主干网络作为核心部件。主干网络是由多个`组件块`组成的一个有向无环图（DAG），框架负责按照DAG的拓扑排序执行个`组件块`关联的代码逻辑，构建TF Graph的一个子图。DAG的输出节点由`concat_blocks`配置项定义，各输出`组件块`的输出tensor拼接之后输入给一个可选的顶部MLP层，或者直接链接到最终的预测层。

![](../../images/component/backbone.jpg)


## 案例1. Wide&Deep 模型

配置文件：[wide_and_deep_backbone_on_movielens.config](https://github.com/alibaba/EasyRec/tree/master/examples/configs/wide_and_deep_backbone_on_movielens.config)

```
model_config: {
  model_class: "RankModel"
  feature_groups: {
    group_name: 'wide'
    feature_names: 'user_id'
    feature_names: 'movie_id'
    feature_names: 'job_id'
    feature_names: 'age'
    feature_names: 'gender'
    feature_names: 'year'
    feature_names: 'genres'
    wide_deep: WIDE
  }
  feature_groups: {
    group_name: 'deep'
    feature_names: 'user_id'
    feature_names: 'movie_id'
    feature_names: 'job_id'
    feature_names: 'age'
    feature_names: 'gender'
    feature_names: 'year'
    feature_names: 'genres'
    wide_deep: DEEP
  }
  backbone {
    blocks {
      name: 'wide'
      input_layer {
        only_output_feature_list: true
      }
    }
    blocks {
      name: 'deep_logit'
      inputs {
        name: 'deep'
      }
      keras_layer {
        class_name: 'MLP'
        mlp {
          hidden_units: [256, 256, 256, 1]
          use_final_bn: false
          final_activation: 'linear'
        }
      }
    }
    blocks {
      name: 'final_logit'
      inputs {
        name: 'wide'
        input_fn: 'lambda x: tf.add_n(x)'
      }
      inputs {
        name: 'deep_logit'
      }
      merge_inputs_into_list: true
      keras_layer {
        class_name: 'Add'
      }
    }
    concat_blocks: 'final_logit'
  }
  rank_model {
    wide_output_dim: 1
    l2_regularization: 1e-4
  }
  embedding_regularization: 1e-4
}
```

MovieLens-1M数据集效果对比：

| Model               | Epoch | AUC    |
| ------------------- | ----- | ------ |
| Wide&Deep           | 1     | 0.8558 |
| Wide&Deep(Backbone) | 1     | 0.8854 |

备注：通过组件化的方式搭建的模型效果比内置的模型效果更好是因为`MLP`组件有更好的初始化方法。

通过protobuf message `backbone` 来定义主干网络，主干网络有多个积木块（`block`）组成，每个`block`代表一个可复用的组件。

- 每个`block`有一个唯一的名字（name），并且有一个或多个输入和输出。
- 每个输入只能是某个`feature group`的name，或者另一个`block`的name。当一个`block`有多个输入时，会自动执行merge操作（输入为list时自动合并，输入为tensor时自动concat）。
- 所有`block`根据输入与输出的关系组成一个有向无环图（DAG），框架自动解析出DAG的拓扑关系，按照拓扑排序执行块所关联的模块。
- 当`block`有多个输出时，返回一个python元组（tuple），下游`block`可以通过自定义的`input_fn`配置一个lambda表达式函数获取元组的某个值。
- 每个`block`关联的模块通常是一个keras layer对象，实现了一个可复用的子网络模块。框架支持加载自定义的keras layer，以及所有系统内置的keras layer。
- 可以为`block`关联一个`input_layer`对输入的`feature group`配置的特征做一些额外的加工，比如执行`batch normalization`、`layer normalization`、`feature dropout`等操作，并且可以指定输出的tensor的格式（2d、3d、list等）。注意：**当`block`关联的模块是`input_layer`时，不能配置任何输入，并且name必须为某个`feature group`的名字**
- 还有一些特殊的`block`关联了一个特殊的模块，包括`lambda layer`、`sequential layers`、`repeated layer`和`recurrent layer`。这些特殊layer分别实现了自定义表达式、顺序执行多个layer、重复执行某个layer、循环执行某个layer的功能。
- DAG的输出节点名由`concat_blocks`配置项指定，配置了多个输出节点时自动执行tensor的concat操作。
- 可以为主干网络配置一个可选的`MLP`模块。

## 案例2：DeepFM 模型

配置文件：[deepfm_backbone_on_movielens.config](https://github.com/alibaba/EasyRec/tree/master/examples/configs/deepfm_backbone_on_movielens.config)

这个Case重点关注下两个特殊的`block`，一个使用了`lambda`表达式配置了一个自定义函数；另一个的加载了一个内置的keras layer [`tf.keras.layers.Add`](https://keras.io/api/layers/merging_layers/add/)。

```
model_config: {
  model_class: 'RankModel'
  feature_groups: {
    group_name: 'wide'
    feature_names: 'user_id'
    feature_names: 'movie_id'
    feature_names: 'job_id'
    feature_names: 'age'
    feature_names: 'gender'
    feature_names: 'year'
    feature_names: 'genres'
    wide_deep: WIDE
  }
  feature_groups: {
    group_name: 'features'
    feature_names: 'user_id'
    feature_names: 'movie_id'
    feature_names: 'job_id'
    feature_names: 'age'
    feature_names: 'gender'
    feature_names: 'year'
    feature_names: 'genres'
    feature_names: 'title'
    wide_deep: DEEP
  }
  backbone {
    blocks {
      name: 'wide_logit'
      inputs {
        name: 'wide'
      }
      lambda {
        expression: 'lambda x: tf.reduce_sum(x, axis=1, keepdims=True)'
      }
    }
    blocks {
      name: 'features'
      input_layer {
        output_2d_tensor_and_feature_list: true
      }
    }
    blocks {
      name: 'fm'
      inputs {
        name: 'features'
        input_fn: 'lambda x: x[1]'
      }
      keras_layer {
        class_name: 'FM'
      }
    }
    blocks {
      name: 'deep'
      inputs {
        name: 'features'
        input_fn: 'lambda x: x[0]'
      }
      keras_layer {
        class_name: 'MLP'
        mlp {
          hidden_units: [256, 128, 64, 1]
          use_final_bn: false
          final_activation: 'linear'
        }
      }
    }
    blocks {
      name: 'add'
      inputs {
        name: 'wide_logit'
      }
      inputs {
        name: 'fm'
      }
      inputs {
        name: 'deep'
      }
      merge_inputs_into_list: true
      keras_layer {
        class_name: 'Add'
      }
    }
    concat_blocks: 'add'
  }
  rank_model {
    l2_regularization: 1e-4
    wide_output_dim: 1
  }
  embedding_regularization: 1e-4
}
```
MovieLens-1M数据集效果对比：

| Model               | Epoch | AUC    |
| ------------------- | ----- | ------ |
| DeepFM              | 1     | 0.8867 |
| DeepFM(Backbone)    | 1     | 0.8872 |

## 案例3：DCN v2 模型

配置文件：[dcn_backbone_on_movielens.config](https://github.com/alibaba/EasyRec/tree/master/examples/configs/dcn_backbone_on_movielens.config)

这个Case重点关注一个特殊的 DCN `block`，用了`recurrent layer`实现了循环调用某个模块多次的效果。
该Case还是在DAG之上添加了顶部MLP模块。

```
model_config: {
  model_class: 'RankModel'
  feature_groups: {
    group_name: 'all'
    feature_names: 'user_id'
    feature_names: 'movie_id'
    feature_names: 'job_id'
    feature_names: 'age'
    feature_names: 'gender'
    feature_names: 'year'
    feature_names: 'genres'
    wide_deep: DEEP
  }
  backbone {
    blocks {
      name: "deep"
      inputs {
        name: 'all'
      }
      keras_layer {
        class_name: 'MLP'
        mlp {
          hidden_units: [256, 128, 64]
        }
      }
    }
    blocks {
      name: "dcn"
      inputs {
        name: 'all'
        input_fn: 'lambda x: [x, x]'
      }
      recurrent {
        num_steps: 3
        fixed_input_index: 0
        keras_layer {
          class_name: 'Cross'
        }
      }
    }
    concat_blocks: ['deep', 'dcn']
    top_mlp {
      hidden_units: [64, 32, 16]
    }
  }
  rank_model {
    l2_regularization: 1e-4
  }
  embedding_regularization: 1e-4
}
```

上述配置对`Cross` Layer循环调用了3次，逻辑上等价于执行如下语句：

```
x1 = Cross()(x0, x0)
x2 = Cross()(x0, x1)
x3 = Cross()(x0, x2)    
```

MovieLens-1M数据集效果对比：

| Model               | Epoch | AUC    |
| ------------------- | ----- | ------ |
| DCN （内置）         | 1     | 0.8576 |
| DCN_v2 （backbone）  | 1     | 0.8770 |

备注：新实现的`Cross`组件对应了参数量更多的v2版本的DCN，而内置的DCN模型对应了v1版本的DCN。

## 案例4：DLRM 模型

配置文件：[dlrm_backbone_on_criteo.config](https://github.com/alibaba/EasyRec/tree/master/examples/configs/dlrm_backbone_on_criteo.config)

```
model_config: {
  model_class: 'RankModel'
  feature_groups: {
    group_name: "dense"
    feature_names: "F1"
    feature_names: "F2"
    ...
    wide_deep:DEEP
  }
  feature_groups: {
    group_name: "sparse"
    feature_names: "C1"
    feature_names: "C2"
    feature_names: "C3"
    ...
    wide_deep:DEEP
  }
  backbone {
    blocks {
      name: 'bottom_mlp'
      inputs {
        name: 'dense'
      }
      keras_layer {
        class_name: 'MLP'
        mlp {
          hidden_units: [64, 32, 16]
        }
      }
    }
    blocks {
      name: 'sparse'
      input_layer {
        output_2d_tensor_and_feature_list: true
      }
    }
    blocks {
      name: 'dot'
      inputs {
        name: 'bottom_mlp'
        input_fn: 'lambda x: [x]'
      }
      inputs {
        name: 'sparse'
        input_fn: 'lambda x: x[1]'
      }
      keras_layer {
        class_name: 'DotInteraction'
      }
    }
    blocks {
      name: 'sparse_2d'
      inputs {
        name: 'sparse'
        input_fn: 'lambda x: x[0]'
      }
    }
    concat_blocks: ['sparse_2d', 'dot']
    top_mlp {
      hidden_units: [256, 128, 64]
    }
  }
  rank_model {
    l2_regularization: 1e-5
  }
  embedding_regularization: 1e-5
}
```

Criteo数据集效果对比：

| Model             | Epoch | AUC     |
| ----------------- | ----- | ------- |
| DLRM              | 1     | 0.79785 |
| DLRM (backbone)   | 1     | 0.7993  |

备注：`DotInteraction` 是新开发的特征两两交叉做内积运算的模块。

## 案例5：为 DLRM 模型添加一个新的数值特征Embedding组件

配置文件：[dlrm_on_criteo_with_periodic.config](https://github.com/alibaba/EasyRec/tree/master/examples/configs/dlrm_on_criteo_with_periodic.config)

与上一个案例相比，多了一个`PeriodicEmbedding` Layer，组件化编程的**灵活性与可扩展性**由此可见一斑。

重点关注一下`PeriodicEmbedding` Layer的参数配置方式，这里并没有使用自定义protobuf message的传参方式，而是采用了内置的`google.protobuf.Struct`对象作为自定义Layer的参数。实际上，该自定义Layer也支持通过自定义message传参。框架提供了一个通用的`Parameter` API 用通用的方式处理两种传参方式。

```
model_config: {
  model_class: 'RankModel'
  feature_groups: {
    group_name: "dense"
    feature_names: "F1"
    feature_names: "F2"
    ...
    wide_deep:DEEP
  }
  feature_groups: {
    group_name: "sparse"
    feature_names: "C1"
    feature_names: "C2"
    ...
    wide_deep:DEEP
  }
  backbone {
    blocks {
      name: 'num_emb'
      inputs {
        name: 'dense'
      }
      keras_layer {
        class_name: 'PeriodicEmbedding'
        st_params {
          fields {
            key: "output_tensor_list"
            value { bool_value: true }
          }
          fields {
            key: "embedding_dim"
            value { number_value: 16 }
          }
          fields {
            key: "sigma"
            value { number_value: 0.005 }
          }
        }
      }
    }
    blocks {
      name: 'sparse'
      input_layer {
        output_2d_tensor_and_feature_list: true
      }
    }
    blocks {
      name: 'dot'
      inputs {
        name: 'num_emb'
        input_fn: 'lambda x: x[1]'
      }
      inputs {
        name: 'sparse'
        input_fn: 'lambda x: x[1]'
      }
      keras_layer {
        class_name: 'DotInteraction'
      }
    }
    blocks {
      name: 'sparse_2d'
      inputs {
        name: 'sparse'
        input_fn: 'lambda x: x[0]'
      }
    }
    blocks {
      name: 'num_emb_2d'
      inputs {
        name: 'num_emb'
        input_fn: 'lambda x: x[0]'
      }
    }
    concat_blocks: ['num_emb_2d', 'dot', 'sparse_2d']
    top_mlp {
      hidden_units: [256, 128, 64]
    }
  }
  rank_model {
    l2_regularization: 1e-5
  }
  embedding_regularization: 1e-5
}
```

Criteo数据集效果对比：

| Model             | Epoch | AUC     |
| ----------------- | ----- | ------- |
| DLRM              | 1     | 0.79785 |
| DLRM (backbone)   | 1     | 0.7993  |
| DLRM (periodic)   | 1     | 0.7998  |

## 案例6：使用内置的keras layer搭建DNN模型

配置文件：[mlp_on_movielens.config](https://github.com/alibaba/EasyRec/tree/master/examples/configs/mlp_on_movielens.config)

该案例只为了演示可以组件化EasyRec可以使用TF内置的原子粒度keras layer作为通用组件，实际上我们已经有了一个自定义的MLP组件，使用会更加方便。

该案例重点关注一个特殊的`sequential block`，这个组件块内可以定义多个串联在一起的layers，前一个layer的输出作为后一个layer的输入。相比定义多个普通`block`的方式，`sequential block`会更加方便。

备注：调用系统内置的keras layer，自能通过`google.proto.Struct`的格式传参。

```
model_config: {
  model_class: "RankModel"
  feature_groups: {
    group_name: 'features'
    feature_names: 'user_id'
    feature_names: 'movie_id'
    feature_names: 'job_id'
    feature_names: 'age'
    feature_names: 'gender'
    feature_names: 'year'
    feature_names: 'genres'
    wide_deep: DEEP
  }
  backbone {
    blocks {
      name: 'mlp'
      inputs {
        name: 'features'
      }
      layers {
        keras_layer {
          class_name: 'Dense'
          st_params {
            fields {
              key: 'units'
              value: { number_value: 256 }
            }
            fields {
              key: 'activation'
              value: { string_value: 'relu' }
            }
          }
        }
      }
      layers {
        keras_layer {
          class_name: 'Dropout'
          st_params {
            fields {
              key: 'rate'
              value: { number_value: 0.5 }
            }
          }
        }
      }
      layers {
        keras_layer {
          class_name: 'Dense'
          st_params {
            fields {
              key: 'units'
              value: { number_value: 256 }
            }
            fields {
              key: 'activation'
              value: { string_value: 'relu' }
            }
          }
        }
      }
      layers {
        keras_layer {
          class_name: 'Dropout'
          st_params {
            fields {
              key: 'rate'
              value: { number_value: 0.5 }
            }
          }
        }
      }
      layers {
        keras_layer {
          class_name: 'Dense'
          st_params {
            fields {
              key: 'units'
              value: { number_value: 1 }
            }
          }
        }
      }
    }
    concat_blocks: 'mlp'
  }
  rank_model {
    l2_regularization: 1e-4
  }
  embedding_regularization: 1e-4
}
```
MovieLens-1M数据集效果：

| Model               | Epoch | AUC    |
| ------------------- | ----- | ------ |
| MLP                 | 1     | 0.8616 |

## 其他案例（FiBiNet & MaskNet）

两个新的模型：

- FiBiNet模型配置文件：[fibinet_on_movielens.config](https://github.com/alibaba/EasyRec/tree/master/examples/configs/fibinet_on_movielens.config)
- MaskNet模型配置文件：[masknet_on_movielens.config](https://github.com/alibaba/EasyRec/tree/master/examples/configs/masknet_on_movielens.config)

MovieLens-1M数据集效果：

| Model               | Epoch | AUC    |
| ------------------- | ----- | ------ |
| MaskNet             | 1     | 0.8872 |
| FibiNet             | 1     | 0.8893 |

# 组件库

## 1.基础组件

|类名|功能|说明|
|---|---|---|
|MLP|多层感知机|支持配置激活函数、初始化方法、Dropout、是否使用BN等|
|Highway|类似残差链接|可用来对预训练embedding做增量微调，来自Highway Network|
|Gate|门控|多个输入的加权求和|
|PeriodicEmbedding|周期激活函数|数值特征Embedding|
|AutoDisEmbedding|自动离散化|数值特征Embedding|

## 2.特征交叉组件

|类名|功能|说明|
|---|---|---|
|FM| 二阶交叉 |DeepFM模型的组件|
|DotInteraction|二阶内积交叉|DLRM模型的组件|
|Cross|bit-wise交叉|DCN v2模型的组件|
|BiLinear|双线性|FiBiNet模型的组件|
|FiBiNet|SENet & BiLinear|FiBiNet模型|

## 3.特征重要度学习组件

|类名|功能|说明|
|---|---|---|
|SENet|  |FiBiNet模型的组件|
|MaskBlock| |MaskNet模型的组件|
|MaskNet|多个串行或并行的MaskBlock|MaskNet模型|

## 4. 序列特征编码组件

|类名|功能|说明|
|---|---|---|
|DIN|target attention|DIN模组的组件|
|BST|transformer|BST模型的组件|

# 如何自定义组件

在 `easy_rec/python/layers/keras` 目录下新建一个`py`文件，也可直接添加到一个已有的文件中。我们建议目标类似的组件定义在同一个文件中，减少文件数量；比如特征交叉的组件都放在`interaction.py`里。

定义一个继承[`tf.keras.layers.Layer`](https://keras.io/api/layers/base_layer/)的组件类，至少实现两个方法：`__init__`、`call`。

```python
def __init__(self, params, name='xxx', **kwargs):
  pass
def call(self, inputs, training=None, **kwargs):
  pass
```

`__init__`方法的第一个参数`params`接受框架传递给当前组件的参数。支持两种参数配置的方式：`google.protobuf.Struct`、自定义的protobuf message对象。params对象封装了对这两种格式的参数的统一读取接口，如下：

- 检查必传参数，缺失时报错退出：
  `params.check_required(['embedding_dim', 'sigma'])`
- 用点操作符读取参数：
  `sigma = params.sigma`；支持连续点操作符，如`params.a.b`：
- 注意数值型参数的类型，`Struct`只支持float类型，整型需要强制转换：
  `embedding_dim = int(params.embedding_dim)`
- 数组类型也需要强制类型转换: `units = list(params.hidden_units)`
- 指定默认值读取，返回值会被强制转换为与默认值同类型：`activation = params.get_or_default('activation', 'relu')`
- 支持嵌套子结构的默认值读取：`params.field.get_or_default('key', def_val)`
- 判断某个参数是否存在：`params.has_field(key)`
- 【不建议，会限定传参方式】获取自定义的proto对象：`params.get_pb_config()` 
- 读写`l2_regularizer`属性：`params.l2_regularizer`，传给Dense层或dense函数。

`call`方法用来实现主要的模块逻辑，其`inputs`参数可以是一个tenor，或者是一个tensor列表。可选的`training`参数用来标识当前是否是训练模型。

最后也是最重要的一点，新开发的Layer需要在`easy_rec.python.layers.keras.__init__.py`文件中导出才能被框架识别为组件库中的一员。例如要导出`blocks.py`文件中的`MLP`类，则需要添加：`from .blocks import MLP`。

FM layer的代码示例：

```python
class FM(tf.keras.layers.Layer):
  """Factorization Machine models pairwise (order-2) feature interactions without linear term and bias.

  References
    - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
  Input shape.
    - List of 2D tensor with shape: ``(batch_size,embedding_size)``.
    - Or a 3D tensor with shape: ``(batch_size,field_size,embedding_size)``
  Output shape
    - 2D tensor with shape: ``(batch_size, 1)``.
  """

  def __init__(self, params, name='fm', **kwargs):
    super(FM, self).__init__(name, **kwargs)
    self.use_variant = params.get_or_default('use_variant', False)

  def call(self, inputs, **kwargs):
    if type(inputs) == list:
      emb_dims = set(map(lambda x: int(x.shape[-1]), inputs))
      if len(emb_dims) != 1:
        dims = ','.join([str(d) for d in emb_dims])
        raise ValueError('all embedding dim must be equal in FM layer:' + dims)
      with tf.name_scope(self.name):
        fea = tf.stack(inputs, axis=1)
    else:
      assert inputs.shape.ndims == 3, 'input of FM layer must be a 3D tensor or a list of 2D tensors'
      fea = inputs

    with tf.name_scope(self.name):
      square_of_sum = tf.square(tf.reduce_sum(fea, axis=1))
      sum_of_square = tf.reduce_sum(tf.square(fea), axis=1)
      cross_term = tf.subtract(square_of_sum, sum_of_square)
      if self.use_variant:
        cross_term = 0.5 * cross_term
      else:
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=-1, keepdims=True)
    return cross_term
```

# 如何搭建模型

`组件块`的搭建主干网络的核心部件，本小节将会介绍`组件块`的类型、功能和配置参数。

`组件块`的protobuf定义如下：

```
message Block {
    required string name = 1;
    // the input names of feature groups or other blocks
    repeated Input inputs = 2;
    optional int32 input_concat_axis = 3 [default = -1];
    optional bool merge_inputs_into_list = 4;
    optional string extra_input_fn = 5;

    // sequential layers
    repeated Layer layers = 6;
    // only take effect when there are no layers
    oneof layer {
        InputLayer input_layer = 101;
        Lambda lambda = 102;
        KerasLayer keras_layer = 103;
        RecurrentLayer recurrent = 104;
        RepeatLayer repeat = 105;
    }
}
```

`组件块`会自动合并多个输入，默认是执行输入tensors按照最后一个维度做拼接(concat)，以下配置项可以改变默认行为：

- `input_concat_axis` 用来指定输入tensors拼接的维度
- `merge_inputs_into_list` 设为true，则把输入合并到一个列表里，不做concat操作

```
message Input {
    required string name = 1;
    optional string input_fn = 2;
}
```

每一路输入可以配置一个可选的`input_fn`，指定一个lambda函数对输入做一些简单的变换。比如，当某路输入是一个列表对象是，可以用`input_fn: 'lambda x: x[1]'`配置项获取列表的第二个元素值作为这一路的输入。

- `extra_input_fn` 是一个可选的配置项，用来对合并后的多路输入结果做一些额外的变换，需要配置成lambda函数的格式。

目前总共有7种类型的`组件块`，分别是`空组件块`、`输入组件块`、`Lambda组件块`、`KerasLayer组件块`、`循环组件块`、`重复组件块`、`序列组件块`。

## 1. 空组件块

当一个`block`不配置任何layer时就称之为`空组件块`，`空组件块`只执行多路输入的Merge操作。

## 2. 输入组件块

`输入组件块`关联一个`input_layer`，获取、加工并返回原始的特征输入。

`输入组件块`比较特殊，它不能配置输入，并且`输入组件块`的名字必须为某个`feature group`的`group_name`。

```
message InputLayer {
    optional bool do_batch_norm = 1;
    optional bool do_layer_norm = 2;
    optional float dropout_rate = 3;
    optional float feature_dropout_rate = 4;
    optional bool only_output_feature_list = 5;
    optional bool only_output_3d_tensor = 6;
    optional bool output_2d_tensor_and_feature_list = 7;
    optional bool output_seq_and_normal_feature = 8;
}
```
输入层的定义如上，配置下说明如下：

- `do_batch_norm` 是否对输入特征做`batch normalization`
- `do_layer_norm` 是否对输入特征做`layer normalization`
- `dropout_rate` 输入层执行dropout的概率，默认不执行dropout
- `feature_dropout_rate` 对特征整体执行dropout的概率，默认不执行
- `only_output_feature_list` 输出list格式的各个特征
- `only_output_3d_tensor` 输出`feature group`对应的一个3d tensor，在`embedding_dim`相同时可配置该项
- `output_2d_tensor_and_feature_list` 是否同时输出2d tensor与特征list
- `output_seq_and_normal_feature` 是否输出(sequence特征, 常规特征）元组

## 3. Lambda组件块

`Lambda组件块`可以配置一个lambda函数，执行一些教简单的操作。

## 4. KerasLayer组件块

`KerasLayer组件块`是最核心的组件块，负责加载、执行组件代码逻辑。

- `class_name`是要加载的Keras Layer的类名，支持加载自定义的类和系统内置的Layer类。
- `st_params`是以`google.protobuf.Struct`对象格式配置的参数；
- 还可以用自定义的protobuf message的格式传递参数给加载的Layer对象。

配置示例：
```
keras_layer {
  class_name: 'MLP'
  mlp {
    hidden_units: [64, 32, 16]
  }
}
      
keras_layer {
  class_name: 'Dropout'
  st_params {
    fields {
      key: 'rate'
      value: { number_value: 0.5 }
    }
  }
}    
```

## 5. 循环组件块

`循环组件块`可以实现类似RNN的循环调用结构，可以执行某个Layer多次，每次执行的输入包含了上一次执行的输出。在[DCN](https://github.com/alibaba/EasyRec/tree/master/examples/configs/dcn_backbone_on_movielens.config)网络中有循环组件块的示例，如下：

```
recurrent {
  num_steps: 3
  fixed_input_index: 0
  keras_layer {
    class_name: 'Cross'
  }
}
```
上述配置对`Cross` Layer循环调用了3次，逻辑上等价于执行如下语句：

```python
x1 = Cross()(x0, x0)
x2 = Cross()(x0, x1)
x3 = Cross()(x0, x2)    
```

- `num_steps` 配置循环执行的次数
- `fixed_input_index` 配置每次执行的多路输入组成的列表中固定不变的元素；比如上述示例中的`x0`
- `keras_layer` 配置需要执行的组件

## 6. 重复组件块

`重复组件块` 可以使用相同的输入重复执行某个组件多次，实现`multi-head`的逻辑。示例如下：

```
repeat {
  num_repeat: 2
  keras_layer {
    class_name: "MaskBlock"
    mask_block {
      output_size: 512
      aggregation_size: 2048
      input_layer_norm: false
    }
  }
}
```

- `num_repeat` 配置重复执行的次数
- `output_concat_axis` 配置多次执行结果tensors的拼接维度，若不配置则输出多次执行结果的列表
- `keras_layer` 配置需要执行的组件

## 7. 序列组件块

`序列组件块`可以依次执行配置的多个Layer，前一个Layer的输出是后一个Layer的输入。`序列组件块`相对于配置多个首尾相连的普通组件块要更加简单。示例如下：

```
blocks {
  name: 'mlp'
  inputs {
    name: 'features'
  }
  layers {
    keras_layer {
      class_name: 'Dense'
      st_params {
        fields {
          key: 'units'
          value: { number_value: 256 }
        }
        
        fields {
          key: 'activation'
          value: { string_value: 'relu' }
        }
      }
    }
  }
  layers {
    keras_layer {
      class_name: 'Dropout'
      st_params {
        fields {
          key: 'rate'
          value: { number_value: 0.5 }
        }
      }
    }
  }
  layers {
    keras_layer {
      class_name: 'Dense'
      st_params {
        fields {
          key: 'units'
          value: { number_value: 1 }
        }
      }
    }
  }
}
```
