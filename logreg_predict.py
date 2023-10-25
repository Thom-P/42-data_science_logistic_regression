import os, sys
import pandas as pd
import numpy as np
from load_csv import ft_load
from logreg_train import h_theta, cost, grad

#REPLACE NANS WITH ZEROS IN TEST SET
def main():
    try:    
        ##load the coeffs in text file
        thetaT = np.loadtxt('weights_theta.txt')

        # data_dir = './datasets'
        data_dir = './fake_datasets'
        
        ## load test set
        df_test = ft_load(os.path.join(data_dir, 'dataset_test.csv'))
        n_test = df_test.shape[0]

        # get list of courses
        courses = [col for col in df_test if df_test[col].dtype == np.float64 and col != 'Hogwarts House'] # nan is float
        #print(courses)

        # define list of houses (sort to keep same order as in train)
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        #print(houses)

        # remove the two with uniform score distribution across the 4 houses (useless)
        courses.remove('Arithmancy')
        courses.remove('Care of Magical Creatures')
        # Also remove one of the two anticorrelated one (redundant info) (Astronomy and Defense against the darks arts)
        courses.remove('Astronomy')
        #print(courses)

        ## make predictions
        x_test = df_test[courses].to_numpy().transpose()
        X_test = np.vstack((np.ones((1, n_test)), x_test))
        y_est_array = h_theta(thetaT, X_test)

        # get estimated house index
        house_ind_est = np.argmax(y_est_array, axis=0)

        # save predictions as csv
        df_prediction = pd.DataFrame([houses[ind] for ind in house_ind_est], columns=['Hogwarts House'])
        df_prediction.to_csv(os.path.join(data_dir, 'houses.csv'), index_label='Index')
        return 0

    except Exception as err:
        print(f'{type(err).__name__}: {err}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
