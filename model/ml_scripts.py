import logging
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os
from init_scripts import init_logger
import visual
import torch
import torch.nn as nn


def run_model(abp_feat):
    # Get Logger
    logger = init_logger('ml_logs')
    logger.info("----------------------------")
    logger.info("Starting ML model (total SYS, test/train splitting 80/20) + plotting")

    # All Data
    abp = read_single_feature_data(f'/features/train_test/tot_abp_{abp_feat}.csv')
    ppg_feats = read_multiple_feature_data('/features/train_test/tot_ppg_feats.csv')

    ppg_train, ppg_test, abp_train, abp_test = train_test_split(
        ppg_feats, abp, test_size=0.2, random_state=42)  # shuffle=False, stratify=None)

    run_multi_linear_regression('SYS from 34 PPG Features', ppg_train, abp_train, ppg_test, abp_test)

    # run_linear_regression('SYS', ppg_sys_train, abp_sys_train, ppg_sys_test, abp_sys_test)
    # run_linear_regression('DIA', ppg_dia_train, abp_dia_train, ppg_dia_test, abp_dia_test)

    # ppg_train = np.column_stack((ppg_sys_train, ppg_dia_train))
    # ppg_test = np.column_stack((ppg_sys_test, ppg_dia_test))
    # run_multi_linear_regression('SYS from SYS+DIA', ppg_train, abp_sys_train, ppg_test, abp_sys_test)
    #
    # run_sv_regression('SYS', ppg_sys_train, abp_sys_train, ppg_sys_test, abp_sys_test)
    # run_sv_regression('DIA', ppg_dia_train, abp_dia_train, ppg_dia_test, abp_dia_test)
    #
    # run_random_forest('SYS', ppg_sys_train, abp_sys_train, ppg_sys_test, abp_sys_test)
    # run_random_forest('DIA', ppg_dia_train, abp_dia_train, ppg_dia_test, abp_dia_test)

    # run_ann('SYS', ppg_sys_train, abp_sys_train, ppg_sys_test, abp_sys_test)
    # run_ann('DIA', ppg_dia_train, abp_dia_train, ppg_dia_test, abp_dia_test)

    # TODO: RNN models : Feedforward/MLPs, LSTMs, GRUs


def read_single_feature_data(path):
    abs_path = os.path.abspath(os.getcwd())
    dest = abs_path + path
    df = pd.read_csv(dest)
    values = df.values[:, 0]
    # values = calculate_assign_median(values, 125, 7)
    return values


def calculate_assign_median(data, fs, window):
    values, median_values = np.array([]), np.array([])
    time_window = window
    for i in range(len(data)):
        time_passed = i / fs
        values = np.append(values, float(data[i]))
        if time_passed >= time_window:
            for j in range(i-len(median_values)):
                median_values = np.append(median_values, np.median(values))
            values = np.array([])
            time_window = time_window + window
        if len(data) - len(median_values) < window * fs and i == len(data) - 1:
            for j in range(i-len(median_values)+1):
                median_values = np.append(median_values, np.median(values))

    return median_values


def read_multiple_feature_data(path):
    abs_path = os.path.abspath(os.getcwd())
    dest = abs_path + path
    df = pd.read_csv(dest)
    values = df.values
    return values


def run_linear_regression(feat, ppg_train, abp_train, ppg_test, abp_test):
    lr_model = LinearRegression()
    mse, mae, r2, bias, loa = fit_predict_evaluate(lr_model, 'Linear Regression - total ' + feat,
                                                   ppg_train, abp_train, ppg_test, abp_test)
    logging.info(f'LR - MSE: {mse:.3f}, MAE: {mae:.3f}, R^2: {r2:.3f},'
                 f'\t\t\t\t\t  Bias: {bias}, LoA: {loa} ({feat})')


def run_multi_linear_regression(feat, ppg_train, abp_train, ppg_test, abp_test):
    mlr_model = LinearRegression()
    mlr_model.fit(ppg_train, abp_train)
    predictions = mlr_model.predict(ppg_test)
    visual.plot_ml_features_line('MLR SYS+DIA', abp_test, predictions)
    mse = mean_squared_error(abp_test, predictions)
    mae = mean_absolute_error(abp_test, predictions)
    r2 = r2_score(abp_test, predictions)
    bias = np.mean(predictions - abp_test)
    loa_l, loa_u = calculate_limits_of_agreement(predictions, abp_test)
    logging.info(f'MLR - MSE: {mse:.3f}, MAE: {mae:.3f}, R^2: {r2:.3f},\n'
                 f'\t\t\t\t\t  Bias: {bias:.3f}, LoA: ({loa_l:.3f}, {loa_u:.3f}) ({feat})')


def run_sv_regression(feat, ppg_train, abp_train, ppg_test, abp_test):
    svr_model = sklearn.svm.SVR(kernel='sigmoid')
    mse, mae, r2, bias, loa = fit_predict_evaluate(svr_model, 'SVR - total ' + feat,
                                                   ppg_train, abp_train, ppg_test, abp_test)
    logging.info(f'SVR (sigmoid) - MSE: {mse:.3f}, MAE: {mae:.3f}, R^2: {r2:.3f},'
                 f'\t\t\t\t\t  Bias: {bias:.3f}, LoA: {loa} ({feat})')


def run_random_forest(feat, ppg_train, abp_train, ppg_test, abp_test):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    mse, mae, r2, bias, loa = fit_predict_evaluate(rf_model, 'Random Forest - total ' + feat,
                                                   ppg_train, abp_train, ppg_test, abp_test)
    logging.info(f'RF - MSE: {mse:.3f}, MAE: {mae:.3f}, R^2: {r2:.3f},'
                 f'\t\t\t\t\t  Bias: {bias:.3f}, LoA: {loa} ({feat})')


def fit_predict_evaluate(model, model_name, ppg_train, abp_train, ppg_test, abp_test):
    model.fit(ppg_train.reshape(-1, 1), abp_train)
    predictions = model.predict(ppg_test.reshape(-1, 1))
    visual.plot_ml_features(model_name, ppg_test, abp_test, ppg_test, predictions)
    mse = mean_squared_error(abp_test, predictions)
    mae = mean_absolute_error(abp_test, predictions)
    r2 = r2_score(abp_test, predictions)
    bias = np.mean(predictions - abp_test)
    loa_l, loa_u = calculate_limits_of_agreement(predictions, abp_test)
    return mse, mae, r2, bias, [loa_u, loa_u]


def calculate_limits_of_agreement(pred, est):
    differences = pred - est
    mean_difference = np.mean(differences)
    sd_difference = np.std(differences, ddof=1)  # ddof=1 for sample standard deviation
    lower_limit = round(mean_difference - 1.96 * sd_difference, 3)
    upper_limit = round(mean_difference + 1.96 * sd_difference, 3)
    return lower_limit, upper_limit


def main():
    run_model('sys')


if __name__ == "__main__":
    main()
