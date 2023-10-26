import os, sys
import pandas as pd
import numpy as np
from logreg_train import h_theta, cost, grad

def main(argv):
    try:    
        ##load the coeffs and scaling params from bin file
        #thetaT = np.loadtxt('weights_theta.txt')

        if len(argv) == 1:
            data_dir = 'datasets'
        else:
            assert len(argv) == 2 and argv[1] == "--fake", "only supported option is --fake to train on fake dataset"
            data_dir = 'fake_datasets'

        with np.load(os.path.join(data_dir, 'weights_theta.npz')) as data:
            thetaT = data['thetaT_arr']
            x_means = data['x_means']
            x_stds = data['x_stds']

        
        ## load test set
        df_test = pd.read_csv(os.path.join(data_dir, 'dataset_test.csv'), index_col=0)
        n_test = df_test.shape[0]

        # get list of courses
        courses = [col for col in df_test if df_test[col].dtype == np.float64 and col != 'Hogwarts House'] # nan is float
        #print(courses)

        # replace missing scores/nans by 0
        #df_test[courses] = df_test[courses].fillna(0)

        # define list of houses (sort to keep same order as in train)
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
        #print(houses)

        # remove the two with uniform score distribution across the 4 houses (useless)
        courses.remove('Arithmancy')
        courses.remove('Care of Magical Creatures')
        # Also remove one of the two anticorrelated one (redundant info) (Astronomy and Defense against the darks arts)
        courses.remove('Astronomy')
        #print(courses)
        n_courses = len(courses)

        ## make predictions
        x_raw = df_test[courses].to_numpy().transpose()
        x_scaled = np.empty(x_raw.shape)
        for i_row in range(n_courses):
            x_scaled[i_row, :] = (x_raw[i_row, :] - x_means[i_row]) / x_stds[i_row]
        #replace here nans with 0
        np.nan_to_num(x_scaled, copy=False)
        X_test = np.vstack((np.ones((1, n_test)), x_scaled))
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
    sys.exit(main(sys.argv))
