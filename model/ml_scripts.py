import logging

import sklearn.linear_model
import sklearn
import pandas as pd
import os
from init_scripts import init_logger


def run_model():
    # Get Logger
    logger = init_logger('ml_logs')
    logger.info("Starting ML model (median sys and ppg estimation with svr)")

    # Training Data
    ppg_sys_train = read_data('/features/training/med_ppg_sys_train.csv')
    ppg_dia_train = read_data('/features/training/med_ppg_dia_train.csv')
    abp_sys_train = read_data('/features/training/med_abp_sys_train.csv')
    abp_dia_train = read_data('/features/training/med_abp_dia_train.csv')

    # Testing Data
    ppg_sys_test = read_data('/features/testing/med_ppg_sys_test.csv')
    ppg_dia_test = read_data('/features/testing/med_ppg_dia_test.csv')
    abp_sys_test = read_data('/features/testing/med_abp_sys_test.csv')
    abp_dia_test = read_data('/features/testing/med_abp_dia_test.csv')

    run_linear_regression('SYS', ppg_sys_train, abp_sys_train, ppg_sys_test, abp_sys_test)
    run_linear_regression('DIA', ppg_dia_train, abp_dia_train, ppg_dia_test, abp_dia_test)

    run_sv_regression('SYS', ppg_sys_train, abp_sys_train, ppg_sys_test, abp_sys_test)
    run_sv_regression('DIA', ppg_dia_train, abp_dia_train, ppg_dia_test, abp_dia_test)


def read_data(path):
    abs_path = os.path.abspath(os.getcwd())
    dest = abs_path + path
    df = pd.read_csv(dest)
    values = df.values
    return values[:, 0]


def run_linear_regression(feat, ppg_train, abp_train, ppg_test, abp_test):
    # Create and Fit model
    lr_model = sklearn.linear_model.LinearRegression()
    lr_model.fit(ppg_train.reshape(-1, 1), abp_train)

    # Make predictions on the test set
    predictions = lr_model.predict(ppg_test.reshape(-1, 1))

    # Evaluate the model
    mse = sklearn.metrics.mean_squared_error(abp_test, predictions)
    logging.info(f'Linear Regression - Mean Squared Error: {mse} ({feat})')


def run_sv_regression(feat, ppg_train, abp_train, ppg_test, abp_test):
    svr_model = sklearn.svm.SVR(kernel='linear')
    svr_model.fit(ppg_train.reshape(-1, 1), abp_train)

    # Make predictions on the test set
    predictions = svr_model.predict(ppg_test.reshape(-1, 1))

    # Evaluate the model
    mse = sklearn.metrics.mean_squared_error(abp_test, predictions)
    logging.info(f'SVR - Mean Squared Error: {mse} ({feat})')


def main():
    run_model()
    # 5: Overall vector creation
    # Create ‘overall’ vectors by concatenating each of the three vectors across all ICU stays
    # Result: three vectors each of length 1200 (i.e. 20 values for 60 ICU stays)

    # 6: Data labelling
    # Create a vector of ICU stays (i.e. a vector of length 1200
    # which contains the ICU stay ID from which each window was obtained).

    # 7: Split data into training and testing

    # 8: Linear regression model creation
    # Use the model to estimate SBP (or DBP) from each SI value in the testing data.
    #   This should produce a vector of estimated SBP (or DBP) values of length 600.
    # Calculate the errors between the estimated and reference SBP (or DBP) values
    #   (using error = estimated - reference).
    # Calculate error statistics for the entire testing dataset.
    #   e.g. mean absolute error, bias (i.e. mean error),
    #   limits of agreement (i.e. 1.96 * standard deviation of errors).


if __name__ == "__main__":
    main()
