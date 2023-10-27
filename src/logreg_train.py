import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_statistics import ft_mean, ft_std


# Sigmoid aka logistic function
def h_theta(thetaT, X):  # sigmoid of thetaT * X (@ for mat multiplication)
    return 1 / (1 + np.exp(-thetaT @ X))


# Cost function
def cost(thetaT, X, y, n_student):
    return -1 / n_student * np.log(h_theta(thetaT, X)) @ y.T +\
        np.log(1 - h_theta(thetaT, X)) @ (1 - y.T)


# Gradient of the cost function
def grad(thetaT, X, y, n_student):
    return 1 / n_student * (h_theta(thetaT, X) - y) @ X.T


def scale_features(df_train, courses, n_features):
    '''Extract and scale features x (scores)'''
    x = df_train[courses].to_numpy().transpose()

    x_means = np.empty(n_features)
    x_stds = np.empty(n_features)
    x_scaled = np.empty(x.shape)
    for i_row in range(n_features):
        x_means[i_row] = ft_mean(x[i_row, :])
        x_stds[i_row] = ft_std(x[i_row, :])
        x_scaled[i_row, :] = (x[i_row, :] - x_means[i_row]) / x_stds[i_row]

    '''
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(x.T)
    ax[0].set_title('Unscaled features')
    ax[0].legend(courses)
    ax[1].plot(x_scaled.T)
    ax[1].set_title('Scaled features')
    ax[1].legend(courses)
    fig.savefig(os.path.join('figures', 'features.png'))
    '''
    # Augment feature matrix to allow for constant weight theta_0
    X = np.vstack((np.ones((1, x_scaled.shape[1])), x_scaled))

    return X, x_means, x_stds


def gradient_descent(X, y_arr, n_features):
    # Initialize 4 thetaT with random coeffs
    thetaT_arr = np.random.rand(4, n_features + 1)
    #  +1 for constant weight theta0
    n_train = X.shape[1]

    alpha = 0.5  # learning rate
    n_iter = 500
    cost_evolution = np.empty((n_iter + 1, 4))

    for h in range(4):
        y = y_arr[h, :].reshape(1, -1) # need reshape to counter auto squeeze
        thetaT = thetaT_arr[h, :].reshape(1, -1)
        cost_evolution[0, h] = cost(thetaT, X, y, n_train).squeeze()
        for i in range(n_iter):
            thetaT = thetaT - alpha * grad(thetaT, X, y, n_train)
            cost_evolution[i + 1, h] = cost(thetaT, X, y, n_train).squeeze()
        thetaT_arr[h, :] = thetaT

    fig, ax = plt.subplots()
    ax.plot(cost_evolution, '-+')
    fig.savefig(os.path.join('figures', 'cost_evolution.png'))

    return thetaT_arr


def main(argv):
    try:
        if len(argv) == 1:
            data_dir = 'datasets'  # train on original train dataset
        else:
            assert len(argv) == 2 and argv[1] == "--valid", \
                "use --valid to train on validation datasets"
            data_dir = 'validation_datasets'
            # train on sub train set from validation folder

        print('Importing train dataset...')
        df = pd.read_csv(os.path.join(data_dir, 'dataset_train.csv'))
        df.head()

        # Get list of all courses
        courses = [col for col in df if df[col].dtype == np.float64]

        # Get list of houses and sort to use same order during prediction
        houses = sorted(df['Hogwarts House'].unique())

        # Remove the two courses with uniform score distribution
        # across the 4 houses (no useful info)
        courses.remove('Arithmancy')
        courses.remove('Care of Magical Creatures')

        # Also remove one of the two anticorrelated courses (redundant info)
        # (Astronomy and Defense against the darks arts)
        courses.remove('Astronomy')

        # Select needed columns from original dataset
        df_select = df[['Hogwarts House'] + courses]

        # Remove students who have nans in at least one of their courses
        n_before = df_select.shape[0]
        df_train = df_select.dropna(axis=0)
        n_train = df_train.shape[0]
        n_features = len(courses)
        print(f'Using {n_features} features/courses: {courses}')
        print(f'Using {n_train} students out of {n_before}:'
              f' {n_before - n_train} have missing data.')

        X, x_means, x_stds = scale_features(df_train, courses, n_features)

        # Extract labels y:
        # 4 houses => 4 classifiers to implement one vs all
        y_arr = np.empty((4, n_train))
        for h, house in enumerate(houses):
            y_arr[h, :] = np.where(df_train['Hogwarts House'] == house, 1., 0.)

        thetaT_arr = gradient_descent(X, y_arr, n_features)

        # Save the coeffs and scaling params in binary file
        np.savez(
            os.path.join(data_dir, 'weights_theta'),
            thetaT_arr=thetaT_arr, x_means=x_means, x_stds=x_stds
        )

        return 0

    except Exception as err:
        print(f'{type(err).__name__}: {err}')
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
