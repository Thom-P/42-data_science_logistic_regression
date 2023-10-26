import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from my_statistics import ft_mean, ft_std


# sigmoid aka logistic function
def h_theta(thetaT, X):  # sigmoid of thetaT * X (@ for mat multiplication)
    return 1 / (1 + np.exp(-thetaT @ X))


# cost function
def cost(thetaT, X, y, n_student):
    return -1 / n_student * np.log(h_theta(thetaT, X)) @ y.T + np.log(1 - h_theta(thetaT, X)) @ (1 - y.T)


# gradient of theta_T
def grad(thetaT, X, y, n_student):
    return 1 / n_student * (h_theta(thetaT, X) - y) @ X.T


def main(argv):
    try:

        random.seed(42) # to keep things reproducible for now
        
        if len(argv) == 1:
            data_dir = 'datasets'
        else:
            assert len(argv) == 2 and argv[1] == "--fake", "only supported option is --fake to train on fake dataset"
            data_dir = 'fake_datasets'
        
        df = pd.read_csv(os.path.join(data_dir, 'dataset_train.csv'))
        # print(df)

        # get list of courses
        courses = [col for col in df if df[col].dtype == np.float64]
        #print(courses)

        # get list of houses and sort to define same order in test 
        houses = sorted(df['Hogwarts House'].unique())
        #print(houses)

        # remove the two with uniform score distribution across the 4 houses (useless)
        courses.remove('Arithmancy')
        courses.remove('Care of Magical Creatures')
        # Also remove one of the two anticorrelated one (redundant info) (Astronomy and Defense against the darks arts)
        courses.remove('Astronomy')
        #print(courses)

        # select needed columns from original dataset
        df_select = df[['Hogwarts House'] + courses]
        #print(df_select)

        # Remove students who have a NaNs in at least one of their course (maybe unnecessary, could try to remove later on)
        df_train = df_select.dropna(axis=0) # this removes 265 students out of 1600
        n_train = df_train.shape[0]
        n_features = len(courses)

        
        ## Extract and scale features x (scores)
        x = df_train[courses].to_numpy().transpose()
        #fig, ax = plt.subplots()
        #ax.plot(x.T)
        #plt.show()

        x_means = np.empty(n_features)
        x_stds = np.empty(n_features)

        x_scaled = np.empty(x.shape)
        for i_row in range(n_features):
            x_means[i_row] = ft_mean(x[i_row, :])
            x_stds[i_row] = ft_std(x[i_row, :])
            x_scaled[i_row, :] = (x[i_row, :] - x_means[i_row]) / x_stds[i_row]

        #fig, ax = plt.subplots()
        #ax.plot(x_scaled.T)
        #plt.show()

        # augment matrix for constant term theta0
        X = np.vstack((np.ones((1, n_train)), x_scaled))

        # Extract labels y: 
        # because 4 houses, need 4 different classifiers to implement one vs all
        # and thus 4 different binary y_labels
        y_arr = np.empty((4, n_train))
        for h, house in enumerate(houses):
            y_arr[h, :] = np.where(df_train['Hogwarts House'] == house, 1., 0.)

        # Initialize 4 thetaT with random coeffs
        thetaT_arr = np.random.rand(4, n_features + 1)  #  +1 for constant term theta0

        # test descent
        alpha = 0.5
        n_iter = 500
        cost_evol_arr = np.empty((n_iter + 1, 4))

        for h in range(4):
            y = y_arr[h, :].reshape(1, -1) # need to reshape to counter auto squeeze
            thetaT = thetaT_arr[h, :].reshape(1, -1)
            cost_evol_arr[0, h] = cost(thetaT, X, y, n_train).squeeze()
            for i in range(n_iter):
                thetaT = thetaT - alpha * grad(thetaT, X, y, n_train)
                cost_evol_arr[i + 1, h] = cost(thetaT, X, y, n_train).squeeze()
            thetaT_arr[h, :] = thetaT

        #fig, ax = plt.subplots()
        #ax.plot(cost_evol_arr, '-+')
        #plt.show()
        
        # create a theta_eff set of weight corrected for feature scaling and allowing for direct use of raw x
        # unused in the end because only work when no missing features in test set
        #x_means_augmented = np.insert(x_means, 0, 0)
        #x_std_augmented = np.insert(x_stds, 0, 1)
        #thetaT_eff = thetaT_arr / x_std_augmented
        #thetaT_eff[:, 0] = thetaT_eff[:, 0] - (thetaT_eff @ x_means_augmented.reshape(-1, 1)).squeeze() # -1 finds the length

        ## save the coeffs and scaling params in binary file
        np.savez(os.path.join(data_dir,'weights_theta'), thetaT_arr=thetaT_arr, x_means=x_means, x_stds=x_stds)

        return 0

    except Exception as err:
        print(f'{type(err).__name__}: {err}')
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
