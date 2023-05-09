# MovieLens-1M

任务：CTR预估/排序

数据集下载：http://files.grouplens.org/datasets/movielens/ml-1m.zip

MovieLens 1M 数据集，包含6000个用户在近4000部电影上的1亿条评论。

数据集分为三个文件：用户数据users.dat，电影数据movies.dat和评分数据ratings.dat。

## 用户数据

分别有用户ID、性别、年龄、职业ID和邮编等字段。

数据中的格式：`UserID::Gender::Age::Occupation::Zip-code`

可以看出UserID、Gender、Age和Occupation都是类别字段，其中邮编字段是我们不使用的。

- Gender is denoted by a "M" for male and "F" for female

- Age is chosen from the following ranges:

  ```
  1: "Under 18"
  18: "18-24"
  25: "25-34"
  35: "35-44"
  45: "45-49"
  50: "50-55"
  56: "56+"
  ```

- Occupation is chosen from the following choices:

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

分别有电影ID、电影名和电影风格等字段。

数据中的格式：`MovieID::Title::Genres`

MovieID是类别字段，Title是文本，Genres也是类别字段

- Titles are identical to titles provided by the IMDB (including year of release)

- Genres are pipe-separated and are selected from the following genres:

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

- UserIDs range between 1 and 6040

- MovieIDs range between 1 and 3952

- Ratings are made on a 5-star scale (whole-star ratings only)

- Timestamp is represented in seconds since the epoch as returned by time(2)

- Each user has at least 20 ratings

# 数据预处理

我们参考了[AutoInt论文](https://dl.acm.org/doi/pdf/10.1145/3357384.3357925)中的处理方法，将评分小于 3 的样本视为负样本，因为低分表示用户不喜欢这部电影；将评分大于 3 的样本视为正样本，并删除中性样本，即评分等于 3。

- label：将评分大于3的作为正样本（label=1），将评分小于3的作为负样本（label=0），进行点击率预估任务。
- UserID、Occupation和MovieID不用变。
- Gender字段：需要将‘F’和‘M’转换成0和1。
- Age字段：要转成7个连续数字0~6。
- Genres字段：无需处理，直接转换为EasyRec的TagFeature
- Title字段：将标题和年份拆开为两个特征，其中标题为SequenceFeature，年份为IDFeature。
