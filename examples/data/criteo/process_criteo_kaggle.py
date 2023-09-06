import pandas as pd

category_features = ['F' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]
target_columns = ['label']
columns = target_columns + dense_features + category_features

data_train = pd.read_csv(
    'criteo_kaggle_display/train.txt', sep='\t', names=columns)

samples_num = data_train.shape[0]
print('samples_num:', samples_num, round(samples_num * 0.9))

train_num = int(round(samples_num * 0.9))
data_train[:train_num].to_csv(
    r'criteo_train_data', index=False, sep='\t', mode='a', header=False)
data_train[train_num:].to_csv(
    r'criteo_test_data', index=False, sep='\t', mode='a', header=False)
print('Done.')
