import sys
import os
import random
import numpy as np
import pandas as pd


def split_dataset(df):
    '''Separate original training data into train
    and test subsets (80/20) for validation'''
    n_total = df.shape[0]
    n_train = round(80 / 100 * n_total)
    n_test = n_total - n_train
    ind_test = set(random.sample(range(n_total), n_test))
    ind_train = set(range(n_total)) - ind_test
    df_test = df.iloc[list(ind_test), :]
    df_train = df.iloc[list(ind_train), :]
    df_test.reset_index(drop=True, inplace=True)
    df_train.reset_index(drop=True, inplace=True)

    # Put houses (expected answers) from new test set in separate df
    df_result = df_test['Hogwarts House'].copy()

    # Remove houses from new test set
    df_test.loc[:, 'Hogwarts House'] = np.nan

    return df_train, df_test, df_result


def save_validation_datasets(validation_dir, df_train, df_test, df_result):
    # Save new validation datasets
    df_train.to_csv(
        os.path.join(validation_dir, 'dataset_train.csv'),
        index_label='Index'
    )
    df_test.to_csv(
        os.path.join(validation_dir, 'dataset_test.csv'),
        index_label='Index'
    )
    df_result.to_csv(os.path.join(
        validation_dir, 'expected_result.csv'), index_label='Index'
    )
    return


def main():
    '''Split original train dataset into subsets for validation'''
    try:
        random.seed(13)  # to keep things reproducible for now
        print('Reading original train dataset...')
        df = pd.read_csv(
            os.path.join('datasets', 'dataset_train.csv'), index_col=0
        )

        validation_dir = 'validation_datasets'
        if not os.path.exists(validation_dir):
            print('Creating validation_datasets directory...')
            os.makedirs(validation_dir)
        print('Splitting the original train dataset into two subsets:'
              ' training (80%) and test(20%)...')
        df_train, df_test, df_result = split_dataset(df)
        print('Saving the new datasets as well as expected test results '
              'in the validation_datasets directory...')
        save_validation_datasets(validation_dir, df_train, df_test, df_result)

    except Exception as err:
        print(f'{type(err).__name__}: {err}')


if __name__ == '__main__':
    sys.exit(main())
