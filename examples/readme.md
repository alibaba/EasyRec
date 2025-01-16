# 介绍

我们准备了一系列的Demo帮助用户快速体验EasyRec的功能，降低使用EasyRec的门槛。

这些Demo包含了在公开数据集上针对不同模型做的多种实验，涵盖了推荐系统中的召回任务和排序任务，主要包括数据集下载、预处理、模型配置、训练及评估等过程。

# 安装EasyRec

我们提供了`本地Anaconda安装`和`Docker镜像启动`两种方式。

## 本地Anaconda安装

Demo实验中使用的环境为 `python=3.6.8` + `tenserflow=1.12.0`

```bash
conda create -n py36_tf12 python=3.6.8
conda activate py36_tf12
pip install tensorflow==1.12.0
```

```bash
git clone https://github.com/alibaba/EasyRec.git
cd EasyRec
bash scripts/init.sh
python setup.py install

```

## Docker镜像启动

### 方法一：拉取已上传的镜像（推荐）

```bash
git clone https://github.com/alibaba/EasyRec.git
cd EasyRec

-- Docker环境可选
(1) `python=3.6.9` + `tenserflow=1.15.5`
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py36-tf1.15-0.7.4
docker run -td --network host -v /local_path/EasyRec:/docker_path/EasyRec mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py36-tf1.15-0.7.4
docker exec -it <CONTAINER_ID> bash


(2) `python=3.8.10` + `tenserflow=2.10.0`
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py38-tf2.10-0.7.4
docker run -td --network host -v /local_path/EasyRec:/docker_path/EasyRec mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py38-tf2.10-0.7.4

docker exec -it <CONTAINER_ID> bash
```

### 方法二：自行构建Docker镜像

```bash
git clone https://github.com/alibaba/EasyRec.git
cd EasyRec

-- Docker环境可选
(1) `python=3.6.9` + `tenserflow=1.15.5`
bash scripts/build_docker.sh
sudo docker run -td --network host -v /local_path:/docker_path mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py36-tf1.15-<easyrec_version>

(2) `python=3.8.10` + `tenserflow=2.10.0`
bash scripts/build_docker_tf210.sh
sudo docker run -td --network host -v /local_path:/docker_path mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec/easyrec:py38-tf2.10-<easyrec_version>

sudo docker exec -it <CONTAINER_ID> bash
```

注：\<easyrec_version>需匹配当前EasyRec版本。

# 准备数据集

在`data/xxx/download_and_process.sh`文件中提供了数据集的下载、解压、数据预处理等步骤，执行完成后会在目录下得到`xxx_train_data`和`xxx_test_data`两个文件。

下面分别是三种常用数据集的下载和预处理：

- MovieLens-1M  (详细见:[data/movielens_1m/](data/movielens_1m/)。 也可跳过预处理，直接通过链接下载处理后的数据集： [movies_train_data](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/movielens_1m/movies_train_data)、[movies_test_data](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/movielens_1m/movies_test_data))

  ```bash
  cd examples/data/movielens_1m
  sh download_and_process.sh
  ```

- Criteo-Research-Kaggle   (详细见:[data/criteo/](data/criteo/)。也可跳过预处理，直接通过链接下载处理后的数据集： [criteo_train_data](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/criteo_kaggle/criteo_train_data)、[criteo_test_data](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/criteo_kaggle/criteo_test_data))

  ```bash
  cd examples/data/criteo
  sh download_and_process.sh
  ```

- Amazon Books   (详细见:[data/amazon_books_data/](data/amazon_books_data/)。也可跳过预处理，直接通过链接直接下载处理后的数据集： [amazon_train_data](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/amazon_books/amazon_train_data)、[amazon_test_data](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/amazon_books/amazon_test_data)、[negative_book_data](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/amazon_books/negative_book_data))

  ```bash
  cd examples/data/amazon_books_data
  sh download_and_process.sh
  ```

<!-- ### Amazon-Books

我们提供了数据集的下载、解压、预处理等步骤，处理完成后会得到**amazon_train_data**和**amazon_test_data**两个文件。

```bash
cd data/amazon_books
sh download_and_process.sh
``` -->

# 示例Config

EasyRec的模型训练和评估都是基于config配置文件的，配置文件采用prototxt格式。在大多数任务中，我们只需要创建config文件就能满足相应的应用。

我们提供了用于demo实验的完整示例config文件，详细见: [configs](configs/)。

**排序任务**

- [wide_and_deep_on_movielens.config](configs/wide_and_deep_on_movielens.config)

- [deepfm_on_movielens.config](configs/deepfm_on_movielens.config)

- [deepfm_backbone_on_movielens.config](configs/deepfm_backbone_on_movielens.config)

- [dcn_on_movielens.config](configs/dcn_on_movielens.config)

- [autoint_on_movielens.config](configs/autoint_on_movielens.config)

- [masknet_on_movielens.config](configs/masknet_on_movielens.config)

- [fibinet_on_movielens.config](configs/fibinet_on_movielens.config)

- [fm_on_criteo.config](configs/fm_on_criteo.config)

- [deepfm_on_criteo.config](configs/deepfm_on_criteo.config)

- [deepfm_backbone_on_criteo.config](configs/deepfm_backbone_on_criteo.config)

**召回任务**

- [dssm_on_books.config](configs/dssm_on_books.config)
- [mind_on_books.config](configs/mind_on_books.config)
- [dssm_on_books_negative_sample.config](configs/dssm_on_books_negative_sample.config)
- [mind_on_books_negative_sample.config](configs/mind_on_books_negative_sample.config)

# 训练及评估

通过指定对应的pipeline_config_path文件即可启动命令训练模型并评估。更多模型可参考[models](../../docs/source/models/)。

### 排序任务 + MovieLens-1M 数据集

在此数据集中, 提供了4个模型上的demo示例（[Wide&Deep](rank_model/wide_and_deep.md) / [DeepFM](rank_model/deepfm.md) / [DCN](rank_model/dcn.md) / [AutoInt](rank_model/din.md)）。

- Wide & Deep

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/wide_and_deep_on_movielens.config `

- DeepFM

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/deepfm_on_movielens.config `

- DCN

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/dcn_on_movielens.config `

- AutoInt

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/autoint_on_movielens.config `

### 排序任务 + Criteo Research Kaggle 数据集

在此数据集中, 提供了2个模型上的demo示例（[FM](rank_model/fm.md) / [DeepFM](rank_model/deepfm.md)）。

- FM

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/fm_on_criteo.config`

- DeepFM

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/deepfm_on_criteo.config`

### 召回任务 + Amazon Books 数据集

在此数据集中, 提供了2个模型及其负采样版的demo示例 [DSSM](match_model/dssm.md) / [MIND](match_model/mind.md) / [DSSM-Negative-Sample](match_model/dssm_negative_sample.md) / [MIND-Negative-Sample](match_model/mind_negative_sample.md) 。

- DSSM

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/dssm_on_books.config `

- MIND

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/mind_on_books.config `

- DSSM with Negative Sample

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/dssm_on_books_negative_sample.config `

- MIND with Negative Sample

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/mind_on_books_negative_sample.config `

#### GPU单机单卡:

```bash
CUDA_VISIBLE_DEVICES=0 python -m easy_rec.python.train_eval --pipeline_config_path *.config
```

- --pipeline_config_path: 训练用的配置文件
- --continue_train: 是否继续训

#### GPU PS训练

- ps跑在CPU上
- master跑在GPU:0上
- worker跑在GPU:1上
- Note: 本地只支持ps, master, worker模式，不支持ps, chief, worker, evaluator模式

```bash
wget https://easyrec.oss-cn-beijing.aliyuncs.com/scripts/train_2gpu.sh
sh train_2gpu.sh *.config
```

<!-- #### CPU训练/评估/导出

不指定CUDA_VISIBLE_DEVICES即可，例如:

```bash
 python -m easy_rec.python.train_eval --pipeline_config_path *.config
``` -->

<!-- 例如，在`movielens-1m`数据集上训练`DeepFM`模型并得到评估结果。

```
python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/deepfm_on_movielens.config
```

更多数据集和模型训练任务的命令参考[rank_model/](rank_model/) 和[match_model/](match_model/)。 -->

# 评估及导出

通过修改pipeline_config_path文件即可评估及导出对应的模型。

- 模型评估

  `python -m easy_rec.python.eval --pipeline_config_path examples/configs/deepfm_on_criteo.config`

- 模型导出

  `python -m easy_rec.python.export --pipeline_config_path examples/configs/deepfm_on_criteo.config --export_dir examples/ckpt/export/deepfm_on_criteo`

# 评估结果

在公开数据集上的demo实验以及评估结果如下，仅供参考。

### 排序模型

- MovieLens-1M

  | Model                | Epoch | AUC    |
  | -------------------- | ----- | ------ |
  | MLP                  | 1     | 0.8616 |
  | Wide&Deep            | 1     | 0.8558 |
  | Wide&Deep(Backbone)  | 1     | 0.8854 |
  | MultiTower(Backbone) | 1     | 0.8814 |
  | DeepFM               | 1     | 0.8867 |
  | DeepFM(Backbone)     | 1     | 0.8872 |
  | DCN                  | 1     | 0.8576 |
  | DCN_v2               | 1     | 0.8770 |
  | AutoInt              | 1     | 0.8513 |
  | MaskNet              | 1     | 0.8872 |
  | FibiNet              | 1     | 0.8893 |

  备注：`MovieLens-1M` 数据集较小，评估指标方差较大，以上结果仅供参考。

- Criteo-Research

  | Model             | Epoch | AUC     |
  | ----------------- | ----- | ------- |
  | FM                | 1     | 0.7577  |
  | DeepFM            | 1     | 0.7970  |
  | DeepFM (backbone) | 1     | 0.7970  |
  | DeepFM (periodic) | 1     | 0.7979  |
  | DeepFM (autodis)  | 1     | 0.7982  |
  | DLRM              | 1     | 0.79785 |
  | DLRM (backbone)   | 1     | 0.7983  |
  | DLRM (senet)      | 1     | 0.7995  |
  | DLRM (standard)   | 1     | 0.7949  |
  | DLRM (autodis)    | 1     | 0.7989  |
  | DLRM (periodic)   | 1     | 0.7998  |

### 召回模型

- Amazon Books Data

  | Model | Epoch | AUC    |
  | ----- | ----- | ------ |
  | DSSM  | 2     | 0.8173 |
  | MIND  | 2     | 0.7511 |

<!-- - Amazon Books Data 负采样版

| Model                | Epoch | Recall@Top1 | Recall@Top10 | Recall@Top100 |
| -------------------- | ----- | ----------- | ------------ | ------------- |
| DSSM_negative_sample | 2     | 0.1241      | 0.6326       | 0.9988        |
| MIND_negative_sample | 2     | 0.0096      | 0.0443       | 0.1994        | -->

注：评估召回模型及负采样版的效果建议参考HitRate指标，具体评估方法见[HitRate评估](https://easyrec.oss-cn-beijing.aliyuncs.com/docs/recall_eval.pdf)。
