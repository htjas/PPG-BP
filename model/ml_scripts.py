import logging
import numpy as np
import sklearn.linear_model
import sklearn
import pandas as pd
import os
from init_scripts import init_logger


def run_model():
    # Get Logger
    logger = init_logger('ml_logs')
    logger.info("Starting ML model (total sys and ppg estimation with chebyshev filter with svr)")

    # Training Data
    ppg_sys_train = read_data('/features/training/tot_ppg_sys_train.csv')
    ppg_dia_train = read_data('/features/training/tot_ppg_dia_train.csv')
    abp_sys_train = read_data('/features/training/tot_abp_sys_train.csv')
    abp_dia_train = read_data('/features/training/tot_abp_dia_train.csv')

    # Testing Data
    ppg_sys_test = read_data('/features/testing/tot_ppg_sys_test.csv')
    ppg_dia_test = read_data('/features/testing/tot_ppg_dia_test.csv')
    abp_sys_test = read_data('/features/testing/tot_abp_sys_test.csv')
    abp_dia_test = read_data('/features/testing/tot_abp_dia_test.csv')

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
    logging.info(f'LR - Mean Squared Error: {mse} ({feat})')
    mse = sklearn.metrics.mean_absolute_error(abp_test, predictions)
    logging.info(f'LR - Mean Absolute Error: {mse} ({feat})')
    bias = np.mean(predictions - abp_test)
    logging.info(f'LR - Bias: {bias} ({feat})')
    loa_l, loa_u = calculate_limits_of_agreement(predictions, abp_test)
    logging.info(f'LR - Limits of Agreement: {loa_l, loa_u} ({feat})')


def run_sv_regression(feat, ppg_train, abp_train, ppg_test, abp_test):
    svr_model = sklearn.svm.SVR(kernel='linear')
    svr_model.fit(ppg_train.reshape(-1, 1), abp_train)

    # Make predictions on the test set
    predictions = svr_model.predict(ppg_test.reshape(-1, 1))

    # Evaluate the model
    mse = sklearn.metrics.mean_squared_error(abp_test, predictions)
    logging.info(f'SVR - Mean Squared Error: {mse} ({feat})')
    mae = sklearn.metrics.mean_absolute_error(abp_test, predictions)
    logging.info(f'SVR - Mean Absolute Error: {mae} ({feat})')
    bias = np.mean(predictions - abp_test)
    logging.info(f'SVR - Bias: {bias} ({feat})')
    loa_l, loa_u = calculate_limits_of_agreement(predictions, abp_test)
    logging.info(f'SVR - Limits of Agreement: {loa_l, loa_u} ({feat})')


def calculate_limits_of_agreement(pred, est):
    differences = pred - est
    mean_difference = np.mean(differences)
    sd_difference = np.std(differences, ddof=1)  # ddof=1 for sample standard deviation
    lower_limit = mean_difference - 1.96 * sd_difference
    upper_limit = mean_difference + 1.96 * sd_difference
    return lower_limit, upper_limit


def main():
    run_model()


if __name__ == "__main__":
    main()
