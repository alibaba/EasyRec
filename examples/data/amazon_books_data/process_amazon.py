import random

import pandas as pd

print('Start reading data...')
title = ['UserID', 'BookID', 'Time']
print('Reading train data...')
train = pd.read_table(
    'AmazonBooksData/book_train.txt',
    sep=',',
    header=None,
    names=title,
    engine='python',
    encoding='ISO-8859-1')
print('Reading test data...')
test = pd.read_table(
    'AmazonBooksData/book_test.txt',
    sep=',',
    header=None,
    names=title,
    engine='python',
    encoding='ISO-8859-1')

print('Start processing train data...')
train_set = []
for userID, hist in train.groupby('UserID'):
  pos_list = hist['BookID'].tolist()

  # generate negative samples randomly
  def gen_neg():
    neg = pos_list[0]
    while neg in pos_list:
      # 1~367982 is the range of book id
      neg = random.randint(1, 367982)
    return neg

  neg_list_1 = [gen_neg() for i in range(len(pos_list))]
  neg_list_2 = [gen_neg() for i in range(len(pos_list))]
  neg_list_3 = [gen_neg() for i in range(len(pos_list))]
  neg_list_4 = [gen_neg() for i in range(len(pos_list))]

  for i in range(1, len(pos_list)):
    # set the max sequence length to 50
    hist = pos_list[:i][-50:]
    hist_str = '|'.join(map(str, hist))
    if i != len(pos_list):
      # for each positive sample, random generate 4 negative samples
      train_set.append((userID, hist_str, pos_list[i], 1))
      train_set.append((userID, hist_str, neg_list_1[i], 0))
      train_set.append((userID, hist_str, neg_list_2[i], 0))
      train_set.append((userID, hist_str, neg_list_3[i], 0))
      train_set.append((userID, hist_str, neg_list_4[i], 0))

random.shuffle(train_set)

print('Start processing test data...')
test_set = []
for userID, hist in test.groupby('UserID'):
  pos_list = hist['BookID'].tolist()

  # generate negative samples randomly
  def gen_neg():
    neg = pos_list[0]
    while neg in pos_list:
      # 1~367982 is the range of book id
      neg = random.randint(1, 367982)
    return neg

  neg_list_1 = [gen_neg() for i in range(len(pos_list))]
  neg_list_2 = [gen_neg() for i in range(len(pos_list))]
  neg_list_3 = [gen_neg() for i in range(len(pos_list))]
  neg_list_4 = [gen_neg() for i in range(len(pos_list))]
  for i in range(1, len(pos_list)):
    # set the max sequence length to 50
    hist = pos_list[:i][-50:]
    hist_str = '|'.join(map(str, hist))
    if i != len(pos_list):
      # for each positive sample, random generate 4 negative samples
      test_set.append((userID, hist_str, pos_list[i], 1))
      test_set.append((userID, hist_str, neg_list_1[i], 0))
      test_set.append((userID, hist_str, neg_list_2[i], 0))
      test_set.append((userID, hist_str, neg_list_3[i], 0))
      test_set.append((userID, hist_str, neg_list_4[i], 0))
random.shuffle(test_set)

train_set_df = pd.DataFrame(train_set)
test_set_df = pd.DataFrame(test_set)

print('Start writing amazon_train_data...')
train_set_df.to_csv(
    r'amazon_train_data', index=False, sep='\t', mode='a', header=False)
print('Start writing amazon_test_data...')
test_set_df.to_csv(
    r'amazon_test_data', index=False, sep='\t', mode='a', header=False)

print('Negative Sampling')
train_book = train[['BookID']].drop_duplicates()
test_book = test[['BookID']].drop_duplicates()
negative_book = pd.concat([train_book, test_book]).drop_duplicates()
df_ones = pd.DataFrame(
    1, index=negative_book.index, columns=negative_book.columns)
negative_book_data = pd.concat([negative_book, df_ones, negative_book], axis=1)
new_header = ['id:int64', 'weight:float', 'feature:string']
negative_book_data.to_csv(
    r'negative_book_data', index=False, sep='\t', mode='a', header=new_header)
print('Done.')
