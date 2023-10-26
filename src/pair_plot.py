import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def make_scatter_plots(house_dfs, course_pairs, ax_array, house_names, colors):
    """Make the scatter plots for every pair of course scores"""
    plot_count = 0
    for course1, course2 in course_pairs:
        ax_curr = ax_array[plot_count]
        legend_flag = False if plot_count % 20 else True  # one legend per fig
        for h, house_df in enumerate(house_dfs):
            house_df.plot.scatter(course1, course2, ax=ax_curr,
                                  c=colors[h % len(colors)],
                                  label=house_names[h],
                                  legend=legend_flag)
        ax_curr.set_title(f'{course1} - {course2}')
        ax_curr.set_xlabel('Score 1')
        ax_curr.set_ylabel('Score 2')
        plot_count += 1


def make_figures(courses, house_names, house_dfs):
    """Prepare 4 figures to draw all the scatter plots in matrix form
    for better visibility"""
    colors = list(mcolors.BASE_COLORS.keys())

    plt.style.use('dark_background')
    figs, axes = zip(*[plt.subplots(5, 4, figsize=(19, 10))
                     for i in range(4)])
    ax_array = np.concatenate(tuple(map(np.ndarray.flatten, axes)))
    course_pairs = [(course1, course2) for i1, course1 in enumerate(courses)
                    for course2 in courses[i1 + 1:]]

    make_scatter_plots(house_dfs, course_pairs, ax_array, house_names, colors)
    ax_array[-1].axis('off')  # turn off unused axes
    ax_array[-2].axis('off')  # turn off unused axes
    for ind, fig in enumerate(figs):
        fig.tight_layout()
        fig.savefig(f'./figures/pair_plot{ind + 1}.png')
    # plt.show()


def main():
    """Output every possible scatter plot of student scores by houses,
    for every possible course pairing"""
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
        make_figures(courses, house_names, house_dfs)
        return 0

    except Exception as err:
        print(f'{type(Exception).__name__}: {err}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
