# EasyRec简介

## What is EasyRec?

![intro.png](docs/images/intro.png)

### EasyRec is an easy to use framework for Recommendation

EasyRec致力于成为容易上手的工业界深度学习推荐算法框架，支持大规模训练、评估、导出和部署。EasyRec实现了业界领先的模型，包含排序、召回、多目标等模型，支持超参搜索，显著降低了建模的复杂度和工作量。

## Why EasyRec?

### Run everywhere

- Local / [MaxCompute](https://help.aliyun.com/product/27797.html) / [DataScience](https://help.aliyun.com/document_detail/170836.html) / [DLC](https://www.alibabacloud.com/help/zh/doc-detail/165137.htm?spm=a2c63.p38356.b99.79.4c0734a4bVav8D)
- TF1.12-1.15 / TF2.x / PAI-TF

### Diversified input data

- [MaxCompute Table](https://help.aliyun.com/document_detail/27819.html?spm=a2c4g.11186623.6.554.91d517bazK7nTF)
- HDFS files
- [OSS files](https://help.aliyun.com/product/31815.html?spm=5176.7933691.1309819.8.5bb52a66ZQOobj)
- Kafka Streams
- Local CSV

### Simple to config

- Flexible feature config and simple model config
- Efficient and robust feature generation\[used in taobao\]
- Nice web interface in development

### It is smart

- EarlyStop / Best Checkpoint Saver
- [Hyper Parameter Search](docs/source/automl/hpo_pai.md) / [AutoFeatureCross](docs/source/automl/auto_cross_emr.md)
- In development: NAS, Knowledge Distillation, MultiModal

### Large scale and easy deployment

- Support large scale embedding, incremental saving
- Many parallel strategies: ParameterServer, Mirrored, MultiWorker
- Easy deployment to [EAS](https://help.aliyun.com/document_detail/113696.html?spm=a2c4g.11174283.6.745.344d1987M3j15E): automatic scaling, easy monitoring
- Consistency guarantee: train and serving

### A variety of models

- [DeepFM](docs/source/models/deepfm.md) / [MultiTower](docs/source/models/multi_tower.md) / [Deep Interest Network](docs/source/models/din.md) / [DSSM](docs/source/models/dssm.md) / [MMoE](docs/source/models/mmoe.md) / [ESMM](docs/source/models/esmm.md)
- More models in development

### Easy to customize

- Easy to implement [customized models](docs/source/models/user_define.md)
- Not need to care about data pipelines

### Get Started

- Download

```
    git clone https://github.com/AlibabaPAI/EasyRec.git
    wget https://easyrec.oss-cn-beijing.aliyuncs.com/data/easyrec_data_20210818.tar.gz
```

- [EasyRec Framework](https://easyrec.oss-cn-beijing.aliyuncs.com/docs/EasyRec.pptx)

- [Run](docs/source/quick_start/local_tutorial.md)

- [PAI-DSW DEMO](https://dsw-dev.data.aliyun.com/#/?fileUrl=http://easyrec.oss-cn-beijing.aliyuncs.com/dsw/easy_rec_demo.ipynb&fileName=EasyRec_DeepFM.ipynb)
  (Rember to select Python 3 kernel)

- [Develop](docs/source/develop.md)

- [Doc](https://easyrec.readthedocs.io/en/latest/)
