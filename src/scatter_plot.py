import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


def make_figure(courses, house_names, house_dfs):
    """Prepare and make the scatter plot of Astronomy vs Defense"""
    plt.style.use('seaborn-v0_8-colorblind')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(10, 5))
    for h, house_df in enumerate(house_dfs):
        house_df.plot.scatter(courses[0], courses[1], ax=ax, c=colors[h],
                              label=house_names[h], s=100)
    ax.set_title(f'{courses[0]} - {courses[1]}')
    ax.set_xlabel('Score 1')
    ax.set_ylabel('Score 2')

    fig.savefig('figures/scatter_plot.png')
    # plt.show()


def main():
    """Create a single scatter plot of the two (anti)correlated features:
    Astronomy and Defense Against the Dark Arts)"""
    try:
        df = pd.read_csv('datasets/dataset_train.csv', index_col=0)

        courses = ('Astronomy', 'Defense Against the Dark Arts')

        # separate data into the 4 houses, and only keep the two courses
        house_names = df['Hogwarts House'].unique()
        house_dfs = [df.loc[df['Hogwarts House'] == name, courses]
                     for name in house_names]

        if not os.path.exists('figures'):
            os.makedirs('figures')
        make_figure(courses, house_names, house_dfs)
        return 0

    except Exception as err:
        print(f'{type(Exception).__name__}: {err}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
