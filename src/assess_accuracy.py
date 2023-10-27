import os
import sys
import pandas as pd


def main():
    try:
        print('Loading prediction results and true houses...')
        df_predict = pd.read_csv(
            os.path.join('validation_datasets', 'houses.csv'),
            index_col=0
        )
        df_truth = pd.read_csv(
            os.path.join('validation_datasets', 'expected_result.csv'),
            index_col=0
        )

        n_test = df_truth.shape[0]
        accuracy = 0
        for i in range(n_test):
            if df_predict.loc[i, 'Hogwarts House']\
                        == df_truth.loc[i, 'Hogwarts House']:
                accuracy += 1
        accuracy /= n_test

        print(f'Accuracy: {accuracy * 100}%')
        return 0

    except Exception as err:
        print(f'{type(err).__name__}: {err}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
