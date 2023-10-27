import os, sys
import pandas as pd
import numpy as np

df_predict = pd.read_csv(os.path.join('validation_datasets', 'houses.csv'), index_col=0)
df_truth = pd.read_csv(os.path.join('validation_datasets', 'expected_result.csv'), index_col=0)

n_test = df_truth.shape[0] 
accuracy = 0
for i in range(n_test):
    if df_predict.loc[i, 'Hogwarts House'] == df_truth.loc[i, 'Hogwarts House']:
        accuracy += 1
accuracy /= n_test

print(f'Accuracy: {accuracy * 100}%')


'''
fig, ax = plt.subplots()
        ax.plot(house_ind_real, '-')
        ax.plot(house_ind_est, 'or')
        plt.show()
## get real house ind
        house_ind_real = np.empty(n_test, dtype='int64')
        for i in range(n_test):
            house_ind_real[i] = list(houses).index(df_test.iloc[i]['Hogwarts House'])
'''
