# benchmark介绍

为了验证算法的准确性、帮助用户更好的使用EasyRec，我们做了大量的benchmark测试。我们还提供公开数据集、EasyRec配置文件，供用户更好的理解和使用EasyRec。

## 单目标数据集

### Taobao 数据集介绍

- 该数据集是淘宝展示广告点击率预估数据集，包含用户、广告特征和行为日志。[天池比赛链接](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56)
- 训练数据表：pai_online_project.easyrec_demo_taobao_train_data
- 测试数据表：pai_online_project.easyrec_demo_taobao_test_data
- 其中pai_online_project是一个公共读的MaxCompute project，里面写入了一些数据表做测试，不需要申请权限。
- 在PAI上面测试使用的资源包括2个parameter server，9个worker，其中一个worker做评估:
  ```json
  {"ps":{"count":2,
         "cpu":1000,
         "memory":40000},
  "worker":{"count":9,
            "cpu":1000,
            "memory":40000}
  }
  ```
- 测试结果

| model      | global_step | best_auc | config                                                                                                        |
| ---------- | ----------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| MultiTower | 1800        | 0.614680 | [taobao_mutiltower.config](http://easyrec.oss-cn-beijing.aliyuncs.com/benchmark/ctr/taobao_mutiltower.config) |
| DIN        | 1600        | 0.617049 | [din.config](http://easyrec.oss-cn-beijing.aliyuncs.com/benchmark/ctr/taobao_din.config)                      |
| DeepFM     | 1600        | 0.580521 | [deepfm.config](http://easyrec.oss-cn-beijing.aliyuncs.com/benchmark/ctr/taobao_deepfm.config)                |
| DCN        | 1500        | 0.596816 | [dcn.config](http://easyrec.oss-cn-beijing.aliyuncs.com/benchmark/ctr/taobao_dcn.config)                      |
| BST        | 3500        | 0.566251 | [bst.config](http://easyrec.oss-cn-beijing.aliyuncs.com/benchmark/ctr/taobao_bst.config)                      |
| AutoInt    | 700         | 0.605982 | [autoint.config](http://easyrec.oss-cn-beijing.aliyuncs.com/benchmark/ctr/taobao_autoint.config)              |

### Avazu CTR 数据集

- 该数据集是DSP广告公司Avazu在Kaggle平台举办的移动广告点击率预测模型挑战赛中使用的。[Click-Through Rate Prediction比赛链接](https://www.kaggle.com/c/avazu-ctr-prediction)
- 训练数据表：pai_online_project.dwd_avazu_ctr_deepmodel_train
- 测试数据表：pai_online_project.dwd_avazu_ctr_deepmodel_test

## 多目标数据集

### AliCCP 数据集

- 数据集采集自手机淘宝移动客户端的推荐系统日志，其中包含点击和与之关联的转化数据。[天池比赛链接](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408)

- 训练数据表：pai_online_project.aliccp_sample_train_kv_split_score

- 测试数据表：pai_online_project.aliccp_sample_test_kv_split_score_1000w (只截取了1000万条)

- 在PAI上面测试使用的资源包括2个parameter server，9个worker，其中一个worker做评估:

  ```json
  {"ps":{"count":2,
         "cpu":1000,
         "memory":40000},
  "worker":{"count":9,
            "cpu":1000,
            "memory":40000}
  }
  ```

- 测试结果

| model           | global_step | ctr auc   | masked cvr auc | ctcvr auc | 训练时间 | config                                                                                                               |
| --------------- | ----------- | --------- | -------------- | --------- | ---- | -------------------------------------------------------------------------------------------------------------------- |
| SimpleMultiTask | 4100        | 0.592606  |                | 0.6306802 | 1小时  | [simple_multi_task.config](http://easyrec.oss-cn-beijing.aliyuncs.com/benchmark/multi_task/simple_multi_task.config) |
| MMoE            | 3100        | 0.5869702 |                | 0.6330008 | 1小时  | [mmoe.config](http://easyrec.oss-cn-beijing.aliyuncs.com/benchmark/multi_task/mmoe.config)                           |
| ESMM            | 800         | 0.5974812 | 0.6841141      | 0.6362526 | 3小时  | [esmm.config](http://easyrec.oss-cn-beijing.aliyuncs.com/benchmark/multi_task/esmm.config)                           |
| PLE             | 3200        | 0.5874    |                | 0.6159    | 2小时  | [ple.config](http://easyrec.oss-cn-beijing.aliyuncs.com/benchmark/multi_task/ple.config)                             |

### CENSUS

- CENSUS有48842个样本数据，每个样本14个属性，包括age, occupation, education, income等。样本的标注值为收入水平，例如>50K、\<=50K。[Census Income数据集链接](https://archive.ics.uci.edu/ml/datasets/census+income)
- 训练数据表：pai_online_project.census_income_train
- 测试数据表：pai_online_project.census_income_test
