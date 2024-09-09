# 知识蒸馏

知识蒸馏在推荐场景中有着广泛的引用，其优势在于不增加模型参数量和特征的情况下也能提高模型的性能。常见应用场景包括:

- 利用精排模型蒸馏粗排模型，可以提升粗排模型的auc，并且使得粗排模型和精排模型有比较好的一致性
- 利用优势特征蒸馏不带该特征的模型, 可以使得模型能够利用优势特征的信息
- 利用gdbt蒸馏DNN模型，可以增强DNN模型对于连续值特征的处理能力

### kd

- loss_name: loss的名称, 默认是'kd_loss\_' + pred_name

- pred_name: 预测的名称, 对于RankModel可以是logits, probs

  - 如果不确定，可以随便填一个，然后在报错信息中，可以查看所有的pred_name

- pred_is_logits: 预测的是logits, 还是probs, 默认是logits

- soft_label_name: 蒸馏的目标, 对应训练数据中的某一列，该目标由teacher模型产生

- label_is_logits: 目标是logits, 还是probs, 默认是logits

- loss_type: loss的类型, 可以是CROSS_ENTROPY_LOSS、L2_LOSS、BINARY_CROSS_ENTROPY_LOSS、KL_DIVERGENCE_LOSS、PAIRWISE_HINGE_LOSS、LISTWISE_RANK_LOSS等

- loss_weight: loss的权重, 默认是1.0

- temperature: 蒸馏的温度，温度越高，student模型学到的细节越丰富, 但对于student模型的能力要求越高, 最优的温度需要通过多次试验才能确定

- Note: 可以设置多个kd, 如多目标场景需要对多个预测结果进行蒸馏

- [示例config](https://easyrec.oss-cn-beijing.aliyuncs.com/configs/dssm_kd_on_taobao.config)

```
data_config {
  input_fields {
    input_name:'clk'
    input_type: INT32
  }

  ...

  input_fields {
    input_name: 'kd_soft'
    input_type: DOUBLE
  }

  label_fields: ['clk', 'kd_soft']
}


model_config {
  model_class: "DSSM"

  ...

  kd {
    soft_label_name: 'kd_soft'
    pred_name: 'logits'
    loss_type: CROSS_ENTROPY_LOSS
    loss_weight: 1.0
    temperature: 2.0
  }
}
```

除了常规的从teacher模型的预测结果里"蒸馏"知识到student模型，在搜推场景中更加推荐采用基于pairwise或者listwise的方式从teacher模型学习
其对不同item的排序（学习对item预估结果的偏序关系），示例如下：

- pairwise 知识蒸馏

```protobuf
  kd {
    loss_name: 'ctcvr_rank_loss'
    soft_label_name: 'pay_logits'
    pred_name: 'logits'
    loss_type: PAIRWISE_HINGE_LOSS
    loss_weight: 1.0
    pairwise_hinge_loss {
      session_name: "raw_query"
      use_exponent: false
      use_label_margin: true
    }
  }
```

- listwise 知识蒸馏

```protobuf
  kd {
    loss_name: 'ctcvr_rank_loss'
    soft_label_name: 'pay_logits'
    pred_name: 'logits'
    loss_type: LISTWISE_RANK_LOSS
    loss_weight: 1.0
    listwise_rank_loss {
      session_name: "raw_query"
      temperature: 3.0
      label_is_logits: true
    }
  }
```

可以为损失函数配置参数，配置方法参考[损失函数](models/loss.md)参数。

### 训练命令

训练命令不改变, 详细参考[模型训练](./train.md)

#### Local

```bash
python -m easy_rec.python.train_eval --pipeline_config_path samples/model_config/dssm_kd_on_taobao.config
```

#### On PAI

```sql
pai -name easy_rec_ext -project algo_public
-Dconfig=oss://easyrec/easy_rec_test/dssm_kd_on_taobao.config
-Dcmd=train
-Dtables=odps://pai_online_project/tables/dwd_avazu_ctr_deepmodel_train,odps://pai_online_project/tables/dwd_avazu_ctr_deepmodel_test
-Dcluster='{"ps":{"count":1, "cpu":1000}, "worker" : {"count":3, "cpu":1000, "gpu":100, "memory":40000}}'
-Darn=acs:ram::xxx:role/ev-ext-test-oss
-Dbuckets=oss://easyrec/
-DossHost=oss-cn-beijing-internal.aliyuncs.com
-Dwith_evaluator=1;
```

#### On EMR

```bash
el_submit -t tensorflow-ps -a easy_rec_train -f dwd_avazu_ctr_deepmodel.config -m local -pn 1 -pc 4 -pm 20000 -wn 3 -wc 6 -wm 20000 -c "python -m easy_rec.python.train_eval --pipeline_config_path dssm_kd_on_taobao.config --continue_train"
```

### 参考文献

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf).

- [Privileged Features Distillation at Taobao Recommendations](https://arxiv.org/pdf/1907.05171.pdf).

- [Knowledge Distillation](https://en.wikipedia.org/wiki/Knowledge_distillation).
