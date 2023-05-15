# Introduction

在召回任务的模型实验中，我们提供了一个公开数据集（Amazon Books）的模型demo。

# Amazon Books 数据集

在此数据集中, 提供了2个模型及其负采样版的demo示例 [DSSM](dssm.md) /  [DSSM-Negative-Sample](dssm_negative_sample.md) / [MIND](mind.md) / [MIND-Negative-Sample](mind_negative_sample.md)。更多模型可参考[models](../../docs/source/models/)。

- DSSM

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/dssm_on_books.config `

- DSSM with Negative Sample

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/dssm_on_books_negative_sample.config `

- MIND

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/mind_on_books.config `

- MIND with Negative Sample

  `python -m easy_rec.python.train_eval --pipeline_config_path examples/configs/mind_on_books_negative_sample.config `

### Results

| DataSet      | Model | Epoch | AUC    |
| ------------ | ----- | ----- | ------ |
| Amazon-Books | DSSM  | 2     | 0.8173 |
| Amazon-Books | MIND  | 2     | 0.7511 |

<!-- | Model                | Epoch | Recall@Top1 | Recall@Top10 | Recall@Top100 |
| -------------------- | ----- | ----------- | ------------ | ------------- |
| DSSM_negative_sample | 2     | 0.1241      | 0.6326       | 0.9988        |
| MIND_negative_sample | 2     | 0.0096      | 0.0443       | 0.1994        | -->

注：评估召回模型及负采样版的效果建议参考HitRate指标，具体评估方法见[HitRate评估](https://easyrec.oss-cn-beijing.aliyuncs.com/docs/recall_eval.pdf)
