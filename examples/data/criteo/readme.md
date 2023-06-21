# Criteo Research Kaggle Display Advertising Challenge Dataset

- 任务：CTR预估/排序

- 简介：

  该数据集由 Criteo 提供，包含数百万展示广告的特征值和点击反馈。 其目的是对点击率 (CTR) 预估的算法进行基准测试。该数据集包括4500万用户的点击记录。有13个连续特征和26个分类特征。

- 下载：

  [官网](https://ailab.criteo.com/ressources/)下载地址：https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz

  天池下载地址：https://tianchi.aliyun.com/dataset/144733

- 详细描述：

  该数据集包含 2 个文件`train.txt` `test.txt`,对应数据的训练和测试部分。

  训练数据集`train.txt`包含7天内Criteo的一部分流量。每行对应Criteo投放的一个展示广告，第一列表示该广告是否被点击。正面（点击）和负面（未点击）的例子都被二次采样（但以不同的速率）以减少数据集的大小。

  有13个采用整数值的特征（主要是计数特征）和26个分类特征。 出于匿名目的，分类特征的值已散列到32位。 这些功能的语义未公开。某些特征可能有缺失值。行按时间顺序排列。

  测试集`test.txt`的计算方式与训练集相同，但它对应于训练期后一天的事件。 第一列（标签）已被删除。

- 格式：

  数据列之间使用制表符作为分隔符：<label> \<integer feature 1> ... \<integer feature 13> \<categorical feature 1> ... \<categorical feature 26>

  当缺少一个值时，该字段只是空的。 测试集中没有标签字段。

# 数据预处理

参考[DeepFM论文](https://arxiv.org/abs/1703.04247)的方式。Criteo数据集包括4500万用户的点击记录。有13个连续特征和26个分类特征。将训练数据集随机分成两部分：90%用于训练，其余10%用于测试。

详细处理细节见 [process_criteo_kaggle.py](process_criteo_kaggle.py)

也可跳过预处理，直接通过链接下载处理后的数据集： [criteo_train_data](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/criteo_kaggle/criteo_train_data)、[criteo_test_data](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/criteo_kaggle/criteo_test_data)。

注：由于测试集中没有标签，无法评估，故在我们的demo实验中没有使用。
