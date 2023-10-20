import sys
import numpy as np
import matplotlib.pyplot as plt
from load_csv import ft_load


def main():
    try:
        df = ft_load('datasets/dataset_train.csv')
        # separate into the 4 houses
        rav = df[df['Hogwarts House'] == 'Ravenclaw']
        sly = df[df['Hogwarts House'] == 'Slytherin']
        gry = df[df['Hogwarts House'] == 'Gryffindor']
        huf = df[df['Hogwarts House'] == 'Hufflepuff']

        fig, ax = plt.subplots(3, 5, figsize=(19, 10))  # size in 100s of pixs
        n_bins = 10
        course_list = [col for col in df if df[col].dtype == np.float64]
        print(course_list)
        print(len(course_list))
        for index, course in enumerate(course_list):
            scores = [rav[course].dropna().to_numpy(),
                      sly[course].dropna().to_numpy(),
                      gry[course].dropna().to_numpy(),
                      huf[course].dropna().to_numpy()]

            ax_curr = ax.flatten()[index]
            ax_curr.hist(scores, n_bins, histtype='bar', stacked=True,
                         label=['Ravenclaw', 'Slytherin',
                                'Gryffindor', 'Hufflepuff'])
            #  non-stacked version
            #  ax_curr.hist(scores, n_bins, histtype='bar')
            ax_curr.set_title(course)
            ax_curr.set_xlabel('Score')
            ax_curr.set_ylabel('# of students')
            if index == 0:
                ax_curr.legend()
        ax[-1, -1].axis('off')  # turn off last two subplots (unused)
        ax[-1, -2].axis('off')
        fig.tight_layout()
        plt.show()
        return 0

    except Exception as err:
        print(f'{type(Exception).__name__}: {err}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
