import logging
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import sklearn
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os
from init_scripts import init_logger
import visual


def run_model():
    # Get Logger
    logger = init_logger('ml_logs')
    logger.info("----------------------------")
    logger.info("Starting ML model (total sys and ppg estimation with Tensorflow ANN) + plotting")

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

    # run_linear_regression('SYS', ppg_sys_train, abp_sys_train, ppg_sys_test, abp_sys_test)
    # run_linear_regression('DIA', ppg_dia_train, abp_dia_train, ppg_dia_test, abp_dia_test)
    #
    # run_sv_regression('SYS', ppg_sys_train, abp_sys_train, ppg_sys_test, abp_sys_test)
    # run_sv_regression('DIA', ppg_dia_train, abp_dia_train, ppg_dia_test, abp_dia_test)
    #
    # run_random_forest('SYS', ppg_sys_train, abp_sys_train, ppg_sys_test, abp_sys_test)
    # run_random_forest('DIA', ppg_dia_train, abp_dia_train, ppg_dia_test, abp_dia_test)

    run_ann('SYS', ppg_sys_train, abp_sys_train, ppg_sys_test, abp_sys_test)
    run_ann('DIA', ppg_dia_train, abp_dia_train, ppg_dia_test, abp_dia_test)


def read_data(path):
    abs_path = os.path.abspath(os.getcwd())
    dest = abs_path + path
    df = pd.read_csv(dest)
    values = df.values
    return values[:, 0]


def run_linear_regression(feat, ppg_train, abp_train, ppg_test, abp_test):
    lr_model = LinearRegression()
    mse, mae, r2, bias, loa = fit_predict_evaluate(lr_model, 'Linear Regression - total ' + feat,
                                                   ppg_train, abp_train, ppg_test, abp_test)
    logging.info(f'LR - MSE: {mse}, MAE: {mae}, R^2: {r2}, Bias: {bias}, LoA: {loa} ({feat})')


def run_sv_regression(feat, ppg_train, abp_train, ppg_test, abp_test):
    svr_model = sklearn.svm.SVR(kernel='sigmoid')
    mse, mae, r2, bias, loa = fit_predict_evaluate(svr_model, 'SVR - total ' + feat,
                                                   ppg_train, abp_train, ppg_test, abp_test)
    logging.info(f'SVR (sigmoid) - MSE: {mse}, MAE: {mae}, R^2: {r2}, Bias: {bias}, LoA: {loa} ({feat})')


def run_random_forest(feat, ppg_train, abp_train, ppg_test, abp_test):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    mse, mae, r2, bias, loa = fit_predict_evaluate(rf_model, 'Random Forest - total ' + feat,
                                                   ppg_train, abp_train, ppg_test, abp_test)
    logging.info(f'RF - MSE: {mse}, MAE: {mae}, R^2: {r2}, Bias: {bias}, LoA: {loa} ({feat})')


def run_ann(feat, x_train, y_train, x_test, y_test):
    # Standardize the data
    # scaler = StandardScaler()
    # x_train_scaled = scaler.fit_transform(x_train)
    # x_test_scaled = scaler.transform(x_test)

    input_shape = x_train.reshape(-1, 1).shape[1]

    # Build a simple neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)  # Output layer with one unit for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(x_train.reshape(-1, 1), y_train, epochs=50, validation_data=(x_test.reshape(-1, 1), y_test))

    # Evaluate the model on the test set
    y_pred = model.predict(x_test.reshape(-1, 1))
    mse = mean_squared_error(y_test, y_pred)

    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

    # Print MSE on the test set
    logging.info(f'TF ANN - MSE: {mse} ({feat})')
    visual.plot_ml_features('ANN ' + feat, x_test, y_test, x_test, y_pred)


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
    lower_limit = mean_difference - 1.96 * sd_difference
    upper_limit = mean_difference + 1.96 * sd_difference
    return lower_limit, upper_limit


def main():
    run_model()


if __name__ == "__main__":
    main()
