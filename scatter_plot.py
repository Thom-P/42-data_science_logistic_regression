import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from load_csv import ft_load


def make_figure(courses, house_names, house_dfs):
    """Prepare and make the scatter plot of Astronomy vs Defense"""
    colors = list(mcolors.BASE_COLORS.keys())

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(19, 10))
    for h, house_df in enumerate(house_dfs):
        house_df.plot.scatter(courses[0], courses[1], ax=ax, c=colors[h],
                              label=house_names[h], s=100, alpha=0.5)
    ax.set_title(f'{courses[0]} - {courses[1]}')
    ax.set_xlabel('Score 1')
    ax.set_ylabel('Score 2')

    fig.savefig('scatter_plot.png')
    # plt.show()


def main():
    """Create a single scatter plot of the two (anti)correlated features:
    Astronomy and Defense Against the Dark Arts)"""
    try:
        df = ft_load('datasets/dataset_train.csv')

        courses = ('Astronomy', 'Defense Against the Dark Arts')

        # separate data into the 4 houses, and only keep the two courses
        house_names = df['Hogwarts House'].unique()
        house_dfs = [df.loc[df['Hogwarts House'] == name, courses]
                     for name in house_names]

        make_figure(courses, house_names, house_dfs)
        return 0

    except Exception as err:
        print(f'{type(Exception).__name__}: {err}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
