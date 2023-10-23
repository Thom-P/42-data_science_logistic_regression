import sys
import os
import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_csv import ft_load
from my_statistics import ft_mean, ft_std

#def main():
#    try:

#random.seed(42) # to keep things reproducible for now
df = ft_load('./datasets/dataset_train.csv')
# print(df)

# get list of courses
courses = [col for col in df if df[col].dtype == np.float64]
#print(courses)

# get list of houses
houses = df['Hogwarts House'].unique()
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
df_clean = df_select.dropna(axis=0) # this removes 265 students out of 1600
n_total = df_clean.shape[0]
n_train = round(80 / 100 * n_total)
n_test = n_total - n_train
n_features = len(courses)

## Separate training data into train and test sets (80/20)
ind_test = set(random.sample(range(n_total), n_test))
ind_train = set(list(range(n_total))) - ind_test
df_test = df_clean.iloc[list(ind_test), :]
df_train = df_clean.iloc[list(ind_train), :]

# equation formulation following coursera course:
# theta_T * X = y
# theta_T is a row vector of weights (transpose of column vec theta)
# X is a m*n matrix for m features (courses) and n data points (students)
# first row should be all 1 (for contanst term theta0)
# y is a row vector of binary value (is or not part of given house)
 
## Extract and scale features x (scores)
x = df_train[courses].to_numpy().transpose()
#print(x.T)
#print(x.shape)
#fig, ax = plt.subplots()
#ax.plot(x.T)
#plt.show()

x_means = np.empty(n_features)
x_stds = np.empty(n_features)

x_scaled = np.empty(x.shape)
for i_row in range(n_features):
    row = x[i_row, :]
    row_mean = ft_mean(row)
    row_std = ft_std(row)
    x_scaled[i_row, :] = (row - row_mean) / row_std
    x_means[i_row] = row_mean
    x_stds[i_row] = row_std

#fig, ax = plt.subplots()
#ax.plot(x_scaled.T)
#plt.show()

X = np.vstack((np.ones((1, n_train)), x_scaled))

# Extract labels y: 
# because 4 houses, need 4 different classifiers to implement one vs all
# and thus 4 different binary y_labels
y_arr = np.empty((4, n_train))
for h, house in enumerate(houses):
    y_arr[h, :] = np.where(df_train['Hogwarts House'] == house, 1., 0.)

# Initialize 4 thetaT with random coeffs
thetaT_arr = np.random.rand(4, n_features + 1)  #  +1 for constant term theta0


# cost function
def cost(thetaT, X, y, n_student):
    return -1 / n_student * np.sum(y * np.log(h_theta(thetaT, X)) + (1 - y) * np.log(1 - h_theta(thetaT, X)), axis=1)


# sigmoid aka logitic function
def h_theta(thetaT, X):  # sigmoid of thetaT * X (@ for mat mult)
    return 1 / (1 + np.exp(-thetaT @ X))


# gradient of theta_T
def grad(thetaT, X, y, n_student):
    return 1 / n_student * (h_theta(thetaT, X) - y) @ X.T


# test descent
alpha = 0.5
n_iter = 500
cost_evol_arr = np.empty((n_iter + 1, 4))

for h in range(4):
    y = y_arr[[h], :] # need extra bracket to preserve dimension
    thetaT = thetaT_arr[[h], :]
    cost_evol_arr[0, h] = cost(thetaT, X, y, n_train)
    for i in range(n_iter):
        thetaT = thetaT - alpha * grad(thetaT, X, y, n_train)
        cost_evol_arr[i + 1, h] = cost(thetaT, X, y, n_train)
    thetaT_arr[h, :] = thetaT

#fig, ax = plt.subplots()
#ax.plot(cost_evol_arr, '-+')
#plt.show()

## check performance on fake test set
x_test = df_test[courses].to_numpy().transpose()

x_test_scaled = np.empty(x_test.shape)
for i_row in range(n_features):
    row = x_test[i_row, :]
    x_test_scaled[i_row, :] = (row - x_means[i_row]) / x_stds[i_row]

X_test = np.vstack((np.ones((1, n_test)), x_test_scaled))

y_est_array = np.empty((len(houses), n_test))
for h in range(4):
    y_est_array[h, :] = h_theta(thetaT_arr[[h], :], X_test)
    #plt.plot(y_est_array[h, :])
#plt.show()
# get estimated house index
house_ind_est = np.argmax(y_est_array, axis=0)

## get real house ind
house_ind_real = np.empty(n_test, dtype='int64')
for i in range(n_test):
    house_ind_real[i] = list(houses).index(df_test.iloc[i]['Hogwarts House'])

#fig, ax = plt.subplots()
#ax.plot(house_ind_real, '-')
#ax.plot(house_ind_est, 'or')
#plt.show()

accuracy = 0
for i in range(n_test):
    if house_ind_real[i] == house_ind_est[i]:
        accuracy += 1
accuracy /= n_test

print(f'Accuracy: {accuracy * 100}%')


#        return 0

#    except Exception as err:
#        print(f'{type(err).__name__}: {err}')
#        return 1


#if __name__ == '__main__':
#    sys.exit(main())