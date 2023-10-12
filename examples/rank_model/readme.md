# Introduction

在排序任务的模型实验中，我们提供了两个公开数据集（MovieLens-1M, Criteo Research Kaggle）的模型demo。

# MovieLens-1M 数据集

在MovieLens-1M 数据集中, 我们提供了4个模型上的demo示例。更多模型可参考[models](../../docs/source/models/)

[Wide&Deep](wide_and_deep.md) / [DeepFM](deepfm.md) / [DCN](dcn.md) / [AutoInt](din.md)

- Wide & Deep

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/wide_and_deep_on_movieslen.config `

- DeepFM

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/deepfm_on_movieslen.config `

- DCN

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/dcn_on_movieslen.config `

- AutoInt

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/autoint_on_movieslen.config `

### Results

| DataSet      | Model     | AUC    |
| ------------ | --------- | ------ |
| MovieLens-1M | Wide&Deep | 0.8558 |
| MovieLens-1M | DeepFM    | 0.8688 |
| MovieLens-1M | DCN       | 0.8576 |
| MovieLens-1M | AutoInt   | 0.8513 |
| MovieLens-1M | MaskNet   | 0.8872 |
| MovieLens-1M | FibiNet   | 0.8879 |

# Criteo Research Kaggle 数据集

在 `Criteo Research Kaggle` 数据集中, 我们提供了2个模型上的demo示例。

[FM](fm.md) / [DeepFM](deepfm.md)

- FM

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/fm_on_criteo.config`

- DeepFM

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/deepfm_on_criteo.config`

### Results

| DataSet         | Model  | AUC    |
| --------------- | ------ | ------ |
| Criteo-Research | FM     | 0.7577 |
| Criteo-Research | DeepFM | 0.7967 |
