import logging
import numpy as np
from matplotlib import pyplot as plt
# from sklearn.linear_model import LinearRegression
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
import torch.optim as optim
import torch.nn.functional as F
import wandb


def run_model(abp_tot_path, ppg_tot_path, abp_med_path, ppg_med_path):
    # Get Logger
    logger = init_logger('ml_logs')
    logger.info("----------------------------")
    logger.info("Starting ML model")
    wandb.init(
        project="ppg-bp"
    )

    # All Data
    abp_tot = read_multiple_feature_data(abp_tot_path)
    ppg_tot = read_multiple_feature_data(ppg_tot_path)
    abp = read_multiple_feature_data(abp_med_path)
    ppg = read_multiple_feature_data(ppg_med_path)
    # abp_sys, abp_dia, abp_map = abp[:, 0], abp[:, 1], abp[:, 2]

    # Linear Regression (PyTorch)
    # ppg_train, ppg_test, abp_train, abp_test = train_test_split(
    #     ppg, abp, test_size=0.2)  # random_state=42, shuffle=False, stratify=None)
    # torch_regression(ppg_train, abp_train, ppg_test, abp_test, 'SYS, DIA, MAP from 34 Median PPG Features')

    # Neural Network / MLP (PyTorch)
    ppg_train, ppg_test, abp_train, abp_test = train_test_split(
        ppg, abp, test_size=0.2)  # random_state=42, shuffle=False, stratify=None)
    torch_neural_net(ppg_train, abp_train, ppg_test, abp_test, 'SYS, DIA, MAP from 34 Median PPG Features')

    # LSTM and GRU (PyTorch)

    #  Support Vector Machine

    # Random Forest

    wandb.finish()

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
            for j in range(i - len(median_values)):
                median_values = np.append(median_values, np.median(values))
            values = np.array([])
            time_window = time_window + window
        if len(data) - len(median_values) < window * fs and i == len(data) - 1:
            for j in range(i - len(median_values) + 1):
                median_values = np.append(median_values, np.median(values))

    return median_values


def read_multiple_feature_data(path):
    abs_path = os.path.abspath(os.getcwd())
    dest = abs_path + path
    df = pd.read_csv(dest)
    values = df.values
    return values


def run_linear_regression(feat, ppg_train, abp_train, ppg_test, abp_test):
    lr_model = sklearn.linear_model.LinearRegression()
    mse, mae, r2, bias, loa = fit_predict_evaluate(lr_model, 'Linear Regression - total ' + feat,
                                                   ppg_train, abp_train, ppg_test, abp_test)
    logging.info(f'LR - MSE: {mse:.3f}, MAE: {mae:.3f}, R^2: {r2:.3f},'
                 f'\t\t\t\t\t  Bias: {bias:.3f}, LoA: {loa} ({feat})')


def run_multi_linear_regression(feat, ppg_train, abp_train, ppg_test, abp_test):
    mlr_model = sklearn.linear_model.LinearRegression()
    mlr_model.fit(ppg_train, abp_train)
    predictions = mlr_model.predict(ppg_test)
    visual.plot_ml_features_line('MLR SYS+DIA', abp_test, predictions)
    mse, mae, r2, bias, loa_l, loa_u = evaluate(abp_test, predictions)
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
    mse, mae, r2, bias, loa_l, loa_u = evaluate(abp_test, predictions)
    return mse, mae, r2, bias, [loa_u, loa_u]


def evaluate(test, pred):
    mse = mean_squared_error(test, pred)
    mae = mean_absolute_error(test, pred)
    r2 = r2_score(test, pred)
    bias = np.mean(pred - test)
    loa_l, loa_u = calculate_limits_of_agreement(pred, test)
    return mse, mae, r2, bias, loa_l, loa_u


def calculate_limits_of_agreement(pred, est):
    differences = pred - est
    mean_difference = np.mean(differences)
    sd_difference = np.std(differences, ddof=1)  # ddof=1 for sample standard deviation
    lower_limit = round(mean_difference - 1.96 * sd_difference, 3)
    upper_limit = round(mean_difference + 1.96 * sd_difference, 3)
    return lower_limit, upper_limit


# Define the linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def torch_regression(X_train, y_train, X_test, y_test, feat):
    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    # y_train = torch.from_numpy(y_train).float().view(-1, 1).to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)
    # y_test = torch.from_numpy(y_test).float().view(-1, 1).to(device)

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    learning_rate = 0.01

    # Instantiate the model, loss function, and optimizer
    model = LinearRegression(input_size=input_size, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Forward pass and loss
        y_predicted = model(X_train)
        loss = criterion(y_predicted, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss})

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print('--------')
    with torch.no_grad():
        # Testing
        y_pred = model(X_test)

        # Converting back to numpy
        y_pred = y_pred.to('cpu').numpy()
        X_test, y_test = X_test.to('cpu').numpy(), y_test.to('cpu').numpy()

        # Evaluating and Logging
        mse, mae, r2, bias, loa_l, loa_u = evaluate(y_test, y_pred)
        rmse = np.sqrt(mse)
        logging.info(f'PyTorch LR: learning_rate={learning_rate}, epochs={num_epochs} ({feat})\n'
                     f'\t\t\t\t\t  MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.3f}, R^2: {r2:.3f}, '
                     f'Bias: {bias:.3f}, LoA: ({loa_l:.3f}, {loa_u:.3f})')

        wandb.log({"mse": mse, "rmse": rmse, "mae": mae})

        # wandb.finish()

        # Plotting
        # visual.plot_ml_features_line('PyTorch LR', y_test, y_pred)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


def torch_neural_net(X_train, y_train, X_test, y_test, feat):
    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    # y_train = torch.from_numpy(y_train).float().view(-1, 1).to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)
    # y_test = torch.from_numpy(y_test).float().view(-1, 1).to(device)

    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    hidden_size = int((input_size + output_size) / 2)
    learning_rate = 0.01

    model = NeuralNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Forward pass and loss
        y_predicted = model(X_train)
        loss = criterion(y_predicted, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss})

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print('--------')
    with torch.no_grad():
        # Testing
        y_pred = model(X_test)

        # Converting back to numpy
        y_pred = y_pred.to('cpu').numpy()
        X_test, y_test = X_test.to('cpu').numpy(), y_test.to('cpu').numpy()

        # Evaluating and Logging
        mse, mae, r2, bias, loa_l, loa_u = evaluate(y_test, y_pred)
        rmse = np.sqrt(mse)
        logging.info(f'PyTorch NN: learning_rate={learning_rate}, epochs={num_epochs} ({feat})\n'
                     f'\t\t\t\t\t  MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.3f}, R^2: {r2:.3f}, '
                     f'Bias: {bias:.3f}, LoA: ({loa_l:.3f}, {loa_u:.3f})')

        wandb.log({"mse": mse, "rmse": rmse, "mae": mae})

        # wandb.finish()

        # Plotting
        # visual.plot_ml_features_line('PyTorch LR', y_test, y_pred)


def main():
    run_model('/features/train_test/tot_abp_feats.csv',
              '/features/train_test/tot_ppg_feats.csv',
              '/features/train_test/med_abp_feats7.csv',
              '/features/train_test/med_ppg_feats7.csv')


if __name__ == "__main__":
    main()
