# Introduction

为了验证算法的准确性、帮助用户更好的使用EasyRec，我们提供了在一些公开数据集上使用EasyRec训练模型的demo实验，供用户更好的理解和使用EasyRec。主要包括数据集下载、预处理、模型配置、训练及评估等过程。

# Install EasyRec

```
git clone https://github.com/alibaba/EasyRec.git
cd EasyRec
bash scripts/init.sh
python setup.py install
```

# Prepare Data

我们提供了数据集的下载、解压、预处理等步骤，处理完成后会得到**xxx_train_data**和**xxx_test_data**两个文件。预处理细节可在[data](data/) 查看。

- MovieLens-1M

  ```
  cd examples/data/movielens_1m
  sh download_and_process.sh
  ```

- Criteo-Research-Kaggle

  ```
  cd examples/data/criteo
  sh download_and_process.sh
  ```

<!-- ### Amazon-Books

我们提供了数据集的下载、解压、预处理等步骤，处理完成后会得到**amazon_train_data**和**amazon_test_data**两个文件。

```
cd data/amazon_books
sh download_and_process.sh
``` -->

# 示例Config

EasyRec的模型训练和评估都是基于config配置文件的，配置文件采用prototxt格式。
我们提供了用于demo实验的完整示例config文件，详细见: [configs](configs/)。

# 训练及评估

通过指定对应的config文件即可启动命令训练模型。

例如，在`movielens-1m`数据集上训练`DeepFM`模型并得到评估结果。

更多模型训练命令参考[rank_model](rank_model/) 和[match_model](match_model/)。

```
python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/deepfm_on_movieslen.config
```

# Results

我们提供了在公开数据集上的demo实验以及评估结果，仅供参考。

<!-- ### Match Model

| DataSet | Model | HitRate |
| ------- | ----- | ------- |
|         | MIND  |         |
|         | DSSM  |         | -->

### Rank Model

- MovieLens-1M

  | Model     | Epoch | AUC    |
  | --------- | ----- | ------ |
  | Wide&Deep | 1     | 0.8558 |
  | DeepFM    | 1     | 0.8688 |
  | DCN       | 1     | 0.8576 |
  | AutoInt   | 1     | 0.8513 |

- Criteo-Research

  | Model  | Epoch | AUC    |
  | ------ | ----- | ------ |
  | FM     | 1     | 0.7577 |
  | DeepFM | 1     | 0.7967 |
