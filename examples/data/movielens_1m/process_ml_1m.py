import re

import pandas as pd
from sklearn.utils import shuffle


def process_data():
  """Load Dataset from File."""
  print('Start processing movielens-1m dataset.')
  # Read user data
  print('----User Data----')
  users_title = ['UserID', 'Gender', 'Age', 'JobID', 'ZipCode']
  users = pd.read_table(
      'ml-1m/users.dat',
      sep='::',
      header=None,
      names=users_title,
      engine='python',
      encoding='ISO-8859-1')
  users = users.filter(regex='UserID|Gender|Age|JobID|ZipCode')
  # process the gender and age of user
  gender_map = {'F': 0, 'M': 1}
  users['Gender'] = users['Gender'].map(gender_map)

  age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
  users['Age'] = users['Age'].map(age_map)

  # read movie data
  print('----Movie Data----')
  movies_title = ['MovieID', 'Title', 'Genres']
  movies = pd.read_table(
      'ml-1m/movies.dat',
      sep='::',
      header=None,
      names=movies_title,
      engine='python',
      encoding='ISO-8859-1')

  # split the title and year in Feature:'Title'
  pattern = re.compile(r'^(.*)\((\d+)\)$')

  title_map = {
      val: pattern.match(val).group(1)
      for ii, val in enumerate(set(movies['Title']))
  }
  year_map = {
      val: pattern.match(val).group(2)
      for ii, val in enumerate(set(movies['Title']))
  }
  movies['Year'] = movies['Title'].map(year_map)
  movies['Title'] = movies['Title'].map(title_map)

  # read rating data
  print('----Rating Data----')
  ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
  ratings = pd.read_table(
      'ml-1m/ratings.dat',
      sep='::',
      header=None,
      names=ratings_title,
      engine='python',
      encoding='ISO-8859-1')
  ratings = ratings.filter(regex='UserID|MovieID|ratings')
  # ratings of 4 and 5 are viewed as positive samples [label:1]
  # ratings of 0, 1 and 2 are viewed as negative samples [label:0]
  # discard samples of rating = 3
  label_map = {1: 0, 2: 0, 3: 2, 4: 1, 5: 1}
  ratings['label'] = ratings['ratings'].map(label_map)

  # concat users, movies and ratings
  data = pd.merge(pd.merge(ratings, users), movies)

  # let field 'label' to postion 1
  new_order = ['label'] + [col for col in data.columns if col != 'label']
  data = data.reindex(columns=new_order)
  # shuffle samples
  data = shuffle(data)
  print('Process Done.')
  return ratings, users, movies, data


ratings, users, movies, data = process_data()
data_new = data[data['label'] < 2]
print(data.count())
print(data_new.count())

# split train set and test set, and write to file
print('Start writing to file.')
data_new[:665110].to_csv(
    r'movies_train_data', index=False, sep='\t', mode='a', header=False)
data_new[665110:].to_csv(
    r'movies_test_data', index=False, sep='\t', mode='a', header=False)
print('Done.')
