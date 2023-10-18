import sys
from ex00_data_analysis.load_csv import load


def main(argv):
    try:
        df = load('../dataset_train.csv')
        for column in df:
            if df[column].dtype != np.float64:
                continue
            df_scoreByHouse = df[column]
            column_np = df[column].dropna().to_numpy()

            if column_np.size == 0:
                continue
            
        return 0
    
    except Exception as err:
        print(f'{type(Exception).__name__}: {err}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
