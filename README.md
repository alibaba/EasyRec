# EasyRec Introduction

&#160;

## What is EasyRec?

![intro.png](docs/images/intro.png)

### EasyRec is an easy to use framework for Recommendation

EasyRec implements state of the art deep learning models used in common recommendation tasks: candidate generation(matching), scoring(ranking), and multi-task learning. It improves the efficiency of generating high performance models by simple configuration and hyper parameter tuning(HPO).

&#160;

## Why EasyRec?

### Run everywhere

- Local / [MaxCompute](https://help.aliyun.com/product/27797.html) / [EMR-DataScience](https://help.aliyun.com/document_detail/170836.html) / [DLC](https://www.alibabacloud.com/help/zh/doc-detail/165137.htm?spm=a2c63.p38356.b99.79.4c0734a4bVav8D)
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
- [Hyper Parameter Search](docs/source/automl/hpo_pai.md) / [AutoFeatureCross](docs/source/automl/auto_cross_emr.md) / [Knowledge Distillation](docs/source/kd.md) / [Features Selection](docs/source/feature/feature.rst#id4)
- In development: NAS / MultiModal

### Large scale and easy deployment

- Support large scale embedding, incremental saving
- Many parallel strategies: ParameterServer, Mirrored, MultiWorker
- Easy deployment to [EAS](https://help.aliyun.com/document_detail/113696.html?spm=a2c4g.11174283.6.745.344d1987M3j15E): automatic scaling, easy monitoring
- Consistency guarantee: train and serving

### A variety of models

- [DSSM](docs/source/models/dssm.md) / [MIND](docs/source/models/mind.md) / [DropoutNet](docs/source/models/dropoutnet.md) / [CoMetricLearningI2I](docs/source/models/co_metric_learning_i2i.md)
- [W&D](docs/source/models/wide_and_deep.md) / [DeepFM](docs/source/models/deepfm.md) / [MultiTower](docs/source/models/multi_tower.md) / [DCN](docs/source/models/dcn.md) / [DIN](docs/source/models/din.md) / [BST](docs/source/models/bst.md)
- [MMoE](docs/source/models/mmoe.md) / [ESMM](docs/source/models/esmm.md) / [DBMTL](docs/source/models/dbmtl.md) / [PLE](docs/source/models/ple.md)
- [CMBF](docs/source/models/cmbf.md) / [UNITER](docs/source/models/uniter.md)
- More models in development

### Easy to customize

- Easy to implement [customized models](docs/source/models/user_define.md)
- Not need to care about data pipelines

### Fast vector retrieve

- Run [knn algorithm](docs/source/vector_retrieve.md) of vectors in distribute environment

&#160;

## Get Started

Running Platform:

- [Local](docs/source/quick_start/local_tutorial.md)
- [MaxCompute](docs/source/quick_start/mc_tutorial.md)
- [EMR-DataScience](docs/source/quick_start/emr_tutorial.md)
- [PAI-DSW (DEMO)](https://dsw-dev.data.aliyun.com/#/?fileUrl=http://easyrec.oss-cn-beijing.aliyuncs.com/dsw/easy_rec_demo.ipynb&fileName=EasyRec_DeepFM.ipynb)

&#160;

## Document

- [Home](https://easyrec.readthedocs.io/en/latest/)
- [FAQ](https://easyrec.readthedocs.io/en/latest/faq.html)
- [EasyRec Framework](https://easyrec.oss-cn-beijing.aliyuncs.com/docs/EasyRec.pptx)(PPT)

&#160;

## Contribute

Any contributions you make are greatly appreciated!

- Please report bugs by submitting a GitHub issue.
- Please submit contributions using pull requests.
- please refer to the [Development](docs/source/develop.md) document for more details.

&#160;

## Contact

### Join Us

- DingDing Group: 32260796. (EasyRec usage general discussion.)

- Email Group: easy_rec@service.aliyun.com.

### Enterprise Service

- If you need EasyRec enterprise service support, or purchase cloud product services, you can contact us by DingDing Group.

&#160;

## License

EasyRec is released under Apache License 2.0. Please note that third-party libraries may not have the same license as EasyRec.
