import sys
import numpy as np
import pandas as pd
from load_csv import ft_load
from my_statistics import ft_count, ft_mean, ft_std, ft_quartile1, \
    ft_quartile3, ft_median, ft_min, ft_max


def main(argv):
    """Load csv into dataframe and compute various statistics on the columns
    containing float numbers. This function simulates the describe() builtin"""
    try:
        assert argv and len(argv) == 2, "need csv dataset as single parameter."
        df = ft_load(argv[1])
        myDict = {'Count': ft_count, 'Mean': ft_mean, 'Std': ft_std,
                  'Min': ft_min, '25%': ft_quartile1, '50%': ft_median,
                  '75%': ft_quartile3, 'Max': ft_max}
        stats_df = pd.DataFrame(index=myDict.keys())
        for column in df:
            if df[column].dtype != np.float64:
                continue
            column_np = df[column].dropna().to_numpy()
            if column_np.size == 0:  # if empty, skip
                continue
            stats_df[column] = [myDict[key](column_np) for key in myDict]
        print(stats_df)
        #  print(df.describe())  #  for comparison with lib equivalent
        return 0

    except Exception as err:
        print(f'{type(err).__name__}: {err}')
        return -1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
