# 介绍

为了帮助用户更快地使用EasyRec，用公开数据集跑通模型训练任务，快速体验EasyRec。

我们提供了在一些公开数据集上使用EasyRec训练模型的demo实验，涵盖了推荐系统中的召回任务和排序任务，主要包括数据集下载、预处理、模型配置、训练及评估等过程。

# 环境准备

Demo实验中使用的环境为 `python=3.6.8` + `tenserflow=1.12.0`

**Anaconda**

```
conda create -n py36_tf12 python=3.6.8
conda activate py36_tf12
pip install tensorflow==1.12.0
```

# 安装EasyRec

```
git clone https://github.com/alibaba/EasyRec.git
cd EasyRec
bash scripts/init.sh
python setup.py install
```

# 准备数据集

在`data/xxx/download_and_process.sh`文件中提供了数据集的下载、解压、数据预处理等步骤，执行完成后会在目录下得到`xxx_train_data`和`xxx_test_data`两个文件。

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

EasyRec的模型训练和评估都是基于config配置文件的，配置文件采用prototxt格式。 我们提供了用于demo实验的完整示例config文件，详细见: [configs/](configs/)。

- wide_and_deep_on_movieslen.config
- deepfm_on_movieslen.config
- dcn_on_movieslen.config
- autoint_on_movieslen.config
- fm_on_criteo.config
- deepfm_on_criteo.config

# 训练及评估

通过指定对应的config文件即可启动命令训练模型。

### MovieLens-1M 数据集

在此数据集中, 提供了4个模型上的demo示例（[Wide&Deep](wide_and_deep.md) / [DeepFM](deepfm.md) / [DCN](dcn.md) / [AutoInt](din.md)）。更多模型可参考[models](../../docs/source/models/)。

- Wide & Deep

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/wide_and_deep_on_movieslen.config `

- DeepFM

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/deepfm_on_movieslen.config `

- DCN

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/dcn_on_movieslen.config `

- AutoInt

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/autoint_on_movieslen.config `

### Criteo Research Kaggle 数据集

在此数据集中, 提供了2个模型上的demo示例（[FM](fm.md) / [DeepFM](deepfm.md)）。

- FM

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/fm_on_criteo.config`

- DeepFM

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/deepfm_on_criteo.config`

<!-- 例如，在`movielens-1m`数据集上训练`DeepFM`模型并得到评估结果。

```
python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/deepfm_on_movieslen.config
```

更多数据集和模型训练任务的命令参考[rank_model/](rank_model/) 和[match_model/](match_model/)。 -->

# 评估结果

在公开数据集上的demo实验以及评估结果如下，仅供参考。

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
