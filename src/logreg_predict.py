import os
import sys
import pandas as pd
import numpy as np
from logreg_train import h_theta


def main(argv):
    try:
        if len(argv) == 1:
            data_dir = 'datasets'
        else:
            assert len(argv) == 2 and argv[1] == "--valid", \
                "use --valid to make prediction on validation dataset"
            data_dir = 'validation_datasets'

        print(f'Loading weights and scaling params from {data_dir}...')
        with np.load(os.path.join(data_dir, 'weights_theta.npz')) as data:
            thetaT = data['thetaT_arr']
            x_means = data['x_means']
            x_stds = data['x_stds']

        # Load test set
        print(f'Loading test set from {data_dir}...')
        df_test = pd.read_csv(os.path.join(data_dir, 'dataset_test.csv'),
                              index_col=0)
        n_test = df_test.shape[0]

        # Get list of courses
        courses = [col for col in df_test if df_test[col].dtype == np.float64
                   and col != 'Hogwarts House']  # nan is float

        # define list of houses (sorted to keep same order as in train)
        houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

        # Remove the three unused features
        courses.remove('Arithmancy')
        courses.remove('Care of Magical Creatures')
        courses.remove('Astronomy')
        n_courses = len(courses)

        # Scale features
        x_raw = df_test[courses].to_numpy().transpose()
        x_scaled = np.empty(x_raw.shape)
        for i_row in range(n_courses):
            x_scaled[i_row, :] = (x_raw[i_row, :] - x_means[i_row])\
                / x_stds[i_row]

        # Replace nan scores with 0
        np.nan_to_num(x_scaled, copy=False)
        X_test = np.vstack((np.ones((1, n_test)), x_scaled))

        # Make predictions
        y_est_array = h_theta(thetaT, X_test)

        # Get best fitting house index
        house_ind_est = np.argmax(y_est_array, axis=0)

        # Save predictions as csv
        df_prediction = pd.DataFrame([houses[ind] for ind in house_ind_est],
                                     columns=['Hogwarts House'])
        print(f'Prediction completed, saving results in {data_dir}/houses.csv...')
        df_prediction.to_csv(os.path.join(data_dir, 'houses.csv'),
                             index_label='Index')
        return 0

    except Exception as err:
        print(f'{type(err).__name__}: {err}')
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
