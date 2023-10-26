import sys
import os
import random
import numpy as np
from load_csv import ft_load

fake_dir = 'fake_datasets'
if not os.path.exists(fake_dir):
    os.makedirs(fake_dir)

random.seed(42) # to keep things reproducible for now
df = ft_load('./datasets/dataset_train.csv')

## Separate training data into train and test sets (80/20)
n_total = df.shape[0]
n_train = round(80 / 100 * n_total)
n_test = n_total - n_train
ind_test = set(random.sample(range(n_total), n_test))
ind_train = set(list(range(n_total))) - ind_test
df_test = df.iloc[list(ind_test), :]
df_train = df.iloc[list(ind_train), :]

## save training dataset 
df_train.reset_index(drop=True, inplace=True)
df_train.to_csv(os.path.join(fake_dir, 'dataset_train.csv'), index_label='Index')

## save houses (expected answers) from test set in separate file
df_result = df_test['Hogwarts House']
df_result.reset_index(drop=True, inplace=True)
#df_result.to_csv(os.path.join(fake_dir, 'expected_result.csv'), index_label='Index')

## remove houses from test set and save
df_test.loc[:, 'Hogwarts House'] = np.nan
df_test.reset_index(drop=True, inplace=True)
df_test.to_csv(os.path.join(fake_dir, 'dataset_test.csv'), index_label='Index')
