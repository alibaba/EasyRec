# 回归模型

回归模型和CTR模型基本一致，只是采用的loss不一样。

如下图所示, 和CTR模型相比增加了:
loss_type: L2_LOSS

## 1. 内置模型

```protobuf
model_config:{
  model_class: "DeepFM"
  feature_groups: {
    group_name: "deep"
    feature_names: "hour"
    feature_names: "c1"
    ...
    feature_names: "site_id_app_id"
    wide_deep:DEEP
  }
  feature_groups: {
    group_name: "wide"
    feature_names: "hour"
    feature_names: "c1"
    ...
    feature_names: "c21"
    wide_deep:WIDE
  }

  deepfm {
    wide_output_dim: 16

    dnn {
      hidden_units: [128, 64, 32]
    }

    final_dnn {
      hidden_units: [128, 64]
    }
    l2_regularization: 1e-5
  }
  embedding_regularization: 1e-7
  loss_type: L2_LOSS
}
```

## 2. 组件化模型

```protobuf
model_config: {
  model_name: 'DeepFM'
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
        feature_group_name: 'wide'
      }
      lambda {
        expression: 'lambda x: tf.reduce_sum(x, axis=1, keepdims=True)'
      }
    }
    blocks {
      name: 'features'
      inputs {
        feature_group_name: 'features'
      }
      input_layer {
        output_2d_tensor_and_feature_list: true
      }
    }
    blocks {
      name: 'fm'
      inputs {
        block_name: 'features'
        input_fn: 'lambda x: x[1]'
      }
      keras_layer {
        class_name: 'FM'
        fm {
          use_variant: true
        }
      }
    }
    blocks {
      name: 'deep'
      inputs {
        block_name: 'features'
        input_fn: 'lambda x: x[0]'
      }
      keras_layer {
        class_name: 'MLP'
        mlp {
          hidden_units: [256, 128, 64]
          use_final_bn: false
          final_activation: 'linear'
        }
      }
    }
    concat_blocks: ['wide_logit', 'fm', 'deep']
    top_mlp {
      hidden_units: [128, 64]
    }
  }
  model_params {
    l2_regularization: 1e-5
    wide_output_dim: 16
  }
  loss_type: L2_LOSS
  embedding_regularization: 1e-4
}
```
