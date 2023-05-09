# Config配置文件说明

## 输入

在我们的demo实验中，采用local环境的csv格式的文件。

```
train_input_path: "examples/data/movielens_1m/movies_train_data"
eval_input_path: "examples/data/movielens_1m/movies_test_data"
model_dir: "examples/ckpt/new_autoint_on_movieslen_ckpt"
```

其中，`train_input_path`是训练集路径，`test_input_path`是测试集路径，`model_dir`是指定模型保存的路径。

## 训练配置

train_config用于配置一些训练时常用的参数，详细见[train.md](../../docs/source/train.md)。

```
train_config {
  log_step_count_steps: 100
  optimizer_config: {
    adam_optimizer: {
      learning_rate: {
        exponential_decay_learning_rate {
          initial_learning_rate: 0.001
          decay_steps: 1000
          decay_factor: 0.5
          min_learning_rate: 0.00001
        }
      }
    }
    use_moving_average: false
  }
  save_checkpoints_steps: 100
  sync_replicas: True
  num_steps: 2500
}
```

## 评估配置

eval_config用于配置训练过程中的评估指标(如AUC)，详细见 [eval.md](../../docs/source/eval.md)。

```
eval_config {
  metrics_set: {
    auc {}
  }
}
```

## 数据配置

data_config用于配置输入文件中各特征列的数据类型，详细见 [data.md](../../docs/source/feature/data.md)。

```
data_config {
  input_fields {
    input_name:'label'
    input_type: INT32
  }
  input_fields {
    input_name:'user_id'
    input_type: INT32
  }
  input_fields {
    input_name:'movie_id'
    input_type: INT32
  }
}
```

## 特征配置

feature_config用于配置特征字段。

```
feature_config: {
  features: {
    input_names: 'user_id'
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 12000
  }
  features: {
    input_names: 'movie_id'
    feature_type: IdFeature
    embedding_dim: 16
    hash_bucket_size: 6000
  }
}
```

## 模型配置

model_config用于配置模型类型以及模型网络具体参数信息等。

```
model_config: {
  model_class: 'DeepFM'
  feature_groups: {
    group_name: 'wide'
    feature_names: 'user_id'
    feature_names: 'movie_id'
    wide_deep: WIDE
  }
  feature_groups: {
    group_name: 'deep'
    feature_names: 'user_id'
    feature_names: 'movie_id'
    wide_deep: DEEP
  }
  deepfm {
    dnn {
      hidden_units: [256, 128, 64]
    }
    l2_regularization: 1e-4
  }
  embedding_regularization: 1e-4
}
```

## 导出配置

export_config用于配置导出模型时的参数，详细见 [export.md](../../docs/source/export.md)。
