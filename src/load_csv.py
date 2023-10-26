import pandas as pd


def ft_load(path: str) -> pd.DataFrame:
    """Load a csv file into a panda dataFrame and returns it"""
    assert len(path) > 0, "empty path."
    try:
        df = pd.read_csv(path, index_col=0)
        print(f'Loading dataset of dimensions {df.shape}')
        return (df)

    except Exception as err:
        print(f'{type(err).__name__}: {err}')
        raise err
