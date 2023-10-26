import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def make_figure(courses, house_names, house_dfs):
    """Prepare and plot the histogram array"""
    colors = list(mcolors.BASE_COLORS.keys())

    plt.style.use('dark_background')
    fig, ax = plt.subplots(3, 5, figsize=(19, 10))  # size in 100s of pixels
    n_bins = 10

    for index, course in enumerate(courses):
        scores = [house_dfs[h][course].dropna().to_numpy()
                  for h in range(len(house_names))]

        ax_curr = ax.flatten()[index]
        #  nb: remove stacked=True for non-stacked version
        ax_curr.hist(scores, n_bins, histtype='bar', stacked=True,
                     label=house_names, color=colors[:len(house_names)])
        ax_curr.set_title(course)
        ax_curr.set_xlabel('Score')
        ax_curr.set_ylabel('# of students')
        if index == 0:
            ax_curr.legend()
    ax[-1, -1].axis('off')  # turn off last two subplots (unused)
    ax[-1, -2].axis('off')
    fig.tight_layout()
    fig.savefig('./figures/histogram.png')
    # plt.show()


def main():
    """Make a histogram array of all course score distributions"""
    try:
        df = pd.read_csv('datasets/dataset_train.csv', index_col=0)

        # get list of courses
        courses = [col for col in df if df[col].dtype == np.float64]

        # separate data into the 4 houses, and only keep courses columns
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
