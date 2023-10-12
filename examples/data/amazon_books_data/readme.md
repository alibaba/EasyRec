# Amazon Books

这是来自亚马逊的大量产品评论抓取数据集。该数据集包含来自约2000万用户的8283万条独立评论。

- 基础描述:

  ```
  Ratings: 82.83 million
  Users:	20.98 million
  Items:	9.35 million
  Timespan:	May 1996 - July 2014
  Metadata
  reviews and ratings
  item-to-item relationships (e.g. "people who bought X also bought Y")
  timestamps
  helpfulness votes
  product image (and CNN features)
  price
  category
  salesRank
  ```

- 下载：
  原始数据集:

  http://jmcauley.ucsd.edu/data/amazon/index.html
  https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1

  ComiRec处理后数据集:

  Tsinghua Cloud: https://cloud.tsinghua.edu.cn/f/e5c4211255bc40cba828/?dl=1
  Dropbox: https://www.dropbox.com/s/m41kahhhx0a5z0u/data.tar.gz?dl=1

# 数据预处理

我们基于[ComiRec](https://github.com/THUDM/ComiRec/tree/master)提供的AmazonBooks数据集进行进一步处理，使其适配EasyRec的召回模型样本格式。

详细处理细节见 [process_amazon.py](process_amazon.py)

也可跳过预处理，直接通过链接下载处理后的数据集： [amazon_train_data](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/amazon_books/amazon_train_data)、[amazon_test_data](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/amazon_books/amazon_test_data)、[negative_book_data](https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/amazon_books/negative_book_data)。

- 序列特征构造：

  为丰富样本特征，充分利用EasyRec处理序列特征的能力，我们对数据集进一步处理。ComiRec数据集中每一行代表一次交互，包含三个字段\<user_id>,\<item_id>,\<time_stamp>。通过对用户进行分组，得到多条序列特征，用'|'分隔。为提高训练效率，我们设定序列特征的最大长度为50。

  例如，

  ```
  user_id book_id time_stamp
  0       a       0
  0       b       1
  0       c       2
  0       d       3

  ---- process ----

  user_id book_id_seq book_id label
  0       a           b       1
  0       a|b         c       1
  0       a|b|c       d       1

  ```

- 负采样:

  原始数据集只包含正样本，为丰富样本，我们进行了随机负采样。对每一条样本，随机负采样4条没有出现在点击序列中的item。

  ```
  user_id book_id_seq book_id
  0       a           b
  0       a|b         c
  0       a|b|c       d

  ---- nagetive sampling ----

  user_id book_id_seq book_id label
  0       a           b       1
  0       a           h       0
  0       a           i       0
  0       a           j       0
  0       a           k       0
  0       a|b         c       1
  0       a|b         l       0
  0       a|b         m       0
  0       a|b         j       0
  0       a|b         o       0
  0       a|b|c       d       1
  0       a|b|c       m       0
  0       a|b|c       j       0
  0       a|b|c       r       0
  0       a|b|c       h       0
  ```
