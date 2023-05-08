## 介绍

为了验证算法的准确性、帮助用户更好的使用EasyRec，我们提供了在一些公开数据集上使用EasyRec训练模型的demo实验，供用户更好的理解和使用EasyRec。主要包括数据集下载、预处理、模型配置、训练及评估等过程。

## 安装 EasyRec

```
git clone https://github.com/alibaba/EasyRec.git
cd EasyRec
bash scripts/init.sh
python setup.py install
```

## 准备数据集

### MovieLens-1M

我们提供了数据集的下载、解压、预处理等步骤，处理完成后会得到**movies_train_data**和**movies_test_data**两个文件。预处理细节可在`movielens_1m/process_ml_1m.py`查看。

```
cd data/movielens_1m
sh download_and_process.sh
```

### Criteo-Research-Kaggle

我们提供了数据集的下载、解压、预处理等步骤，处理完成后会得到**criteo_train_data**和**criteo_test_data**两个文件。预处理细节可在`criteo/process_criteo_kaggle.py`查看。

```
cd data/criteo
sh download_and_process.sh
```

### Amazon-Books

我们提供了数据集的下载、解压、预处理等步骤，处理完成后会得到**amazon_train_data**和**amazon_test_data**两个文件。

```
cd data/amazon_books
sh download_and_process.sh
```

## Config

EasyRec的模型训练和评估都是基于config配置文件的，配置文件采用prototxt格式。
我们提供了用于demo实验的完整config文件，详细见: \[\]

### 输入

在我们的demo实验中，采用local环境的csv格式的文件。

```
train_input_path: "examples/data/movielens_1m/movies_train_data"
eval_input_path: "examples/data/movielens_1m/movies_test_data"
model_dir: "examples/ckpt/new_autoint_on_movieslen_ckpt"
```

其中，`train_input_path`是训练集路径，`test_input_path`是测试集路径，`model_dir`是指定模型保存的路径。

### 训练配置

train_config用于配置一些训练时常用的参数，详细见 `[docs/source/train.md]`

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

### 评估配置

eval_config用于配置训练过程中的评估指标，详细见: \[\]

```
eval_config {
  metrics_set: {
    auc {}
  }
}
```

### 数据配置

data_config用于配置输入文件中各特征列的数据类型。详细见: \[\]

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

### 特征配置

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

### 模型配置

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

### 导出配置

export_config用于配置导出模型时的参数。

```
export_config {
  multi_placeholder: false
}
```

## Train & Eval Model

通过指定对应的config文件即可启动命令训练模型。例如，在movielens-1m数据集上训练DeepFM模型并得到评估结果。

```
python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/deepfm_on_movieslen.config
```

## Demo Results

我们提供了在公开数据集上的demo实验以及评估结果，仅供参考，详细见rank_model和match_model。

### Match Model

| DataSet | Model | HitRate |
| ------- | ----- | ------- |
|         | MIND  |         |
|         | DSSM  |         |

### Rank Model

| DataSet         | Model     | AUC    |
| --------------- | --------- | ------ |
| MovieLens-1M    | Wide&Deep | 0.8558 |
| MovieLens-1M    | DeepFM    | 0.8688 |
| MovieLens-1M    | DCN       | 0.8576 |
| MovieLens-1M    | AutoInt   | 0.8513 |
| Criteo-Research | FM        | 0.7577 |
| Criteo-Research | DeepFM    | 0.7967 |
| AmazonBooks     | DeepFM    |        |
| AmazonBooks     | DIN       |        |
