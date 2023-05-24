# MovieLens-1M

任务：CTR预估/排序

数据集下载：http://files.grouplens.org/datasets/movielens/ml-1m.zip

MovieLens 1M 数据集，包含6000个用户在近4000部电影上的1亿条评论。

数据集分为三个文件：用户数据users.dat，电影数据movies.dat和评分数据ratings.dat。

## 用户数据

分别有用户ID、性别、年龄、职业ID和邮编等字段。

数据中的格式：`UserID::Gender::Age::Occupation::Zip-code`

可以看出UserID、Gender、Age和Occupation都是类别字段，其中邮编字段是我们不使用的。

- 性别用“M”表示男性，“F”表示女性

- 年龄来自以下范围：

  ```
  1: "Under 18"
  18: "18-24"
  25: "25-34"
  35: "35-44"
  45: "45-49"
  50: "50-55"
  56: "56+"
  ```

- 职业包含以下几种：

  ```
  0: "other" or not specified
  1: "academic/educator"
  2: "artist"
  3: "clerical/admin"
  4: "college/grad student"
  5: "customer service"
  6: "doctor/health care"
  7: "executive/managerial"
  8: "farmer"
  9: "homemaker"
  10: "K-12 student"
  11: "lawyer"
  12: "programmer"
  13: "retired"
  14: "sales/marketing"
  15: "scientist"
  16: "self-employed"
  17: "technician/engineer"
  18: "tradesman/craftsman"
  19: "unemployed"
  20: "writer"
  ```

## 电影数据

分别有电影ID、电影标题和电影风格等字段。

数据中的格式：`MovieID::Title::Genres`

MovieID是类别字段，Title是文本，Genres也是类别字段

- 标题与 IMDB 提供的标题相同（包括发行年份）

- 电影风格类型有以下几种：

  ```
  Action
  Adventure
  Animation
  Children's
  Comedy
  Crime
  Documentary
  Drama
  Fantasy
  Film-Noir
  Horror
  Musical
  Mystery
  Romance
  Sci-Fi
  Thriller
  War
  Western
  ```

## 评分数据

分别有用户ID、电影ID、评分和时间戳等字段。

数据中的格式：`UserID::MovieID::Rating::Timestamp`

评分字段Rating就是我们要学习的label，时间戳字段我们不使用。

- UserIDs 范围在 1 到 6040 之间

- MovieIDs 范围在 1 到 3952 之间

- 评级采用 5 星制（仅限全星评级）

- 时间戳以 time(2) 返回的纪元以来的秒数表示

- 每个用户至少有 20 个评分

# 数据预处理

我们参考了[AutoInt论文](https://dl.acm.org/doi/pdf/10.1145/3357384.3357925)中的处理方法，将评分小于 3 的样本视为负样本，因为低分表示用户不喜欢这部电影；将评分大于 3 的样本视为正样本；最后删除中性样本，即评分等于 3。

详细处理细节见 [process_ml_1m.py](process_ml_1m.py)

也可跳过预处理，直接通过链接下载处理后的数据集： [movies_train_data](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/movielens_1m/movies_train_data)、[movies_test_data](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/movielens_1m/movies_test_data)。

- label：将评分大于3的作为正样本（label=1），将评分小于3的作为负样本（label=0），作为点击率预估任务的目标。
- UserID、Occupation和MovieID不用变。
- Gender字段：将‘F’和‘M’变换成0和1。
- Age字段：把年龄离散化为0-6之间的数字（共7个数字）。
- Genres字段：无需处理，直接转换为EasyRec的TagFeature
- Title字段：将标题和年份拆开为两个特征，其中标题为SequenceFeature，年份为IDFeature。
