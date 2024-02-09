import logging
from operator import itemgetter
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


def train_test_model(model_name, f_indexes_names, iteration):
    # Get Logger
    logger = init_logger('ml_logs')
    logger.info("----------------------------")
    logger.info("Starting ML model")
    wandb.init(project="ppg-bp", name=model_name)

    # Data reading
    abp_tot_path, ppg_tot_path, abp_med_path, ppg_med_path = ('/features/train_test/tot_abp.csv',
                                                              '/features/train_test/tot_ppg.csv',
                                                              '/features/train_test/med7_abp.csv',
                                                              '/features/train_test/med7_ppg.csv')
    # abp_tot = read_multiple_feature_data(abp_tot_path)
    # ppg_tot = read_multiple_feature_data(ppg_tot_path)
    abp = read_multiple_feature_data(abp_med_path)
    ppg = read_multiple_feature_data(ppg_med_path)

    # Feature Splicing
    f_indexes = f_indexes_names[:, 0].astype(int)
    ppg = ppg[:, f_indexes]

    # Train/Test splitting
    X_train, X_test, y_train, y_test = train_test_split(
        ppg, abp, test_size=0.2)  # random_state=42, shuffle=False, stratify=None)

    # PyTorch GPU activation
    device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)

    # Data Normalization Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    feat = 'SYS, DIA, MAP from 34 Median PPG Features'

    match model_name:
        case 'LR':
            # Linear Regression (PyTorch)
            torch_regression(model_name, device, X_train, y_train, X_test, y_test, 0.01, 1000, feat)
        case 'MLP':
            # Neural Network / MLP (PyTorch)
            torch_neural_net(model_name, device, X_train, y_train, X_test, y_test, 0.01, 1000, feat)
        case 'LSTM':
            # Splitting data to shorter arrays, to avoid GPU overclocking
            no_segments = 3
            X_train_segments = split_into_segments(X_train, no_segments)
            # X_test_segments = split_into_segments(X_test, no_segments)
            y_train_segments = split_into_segments(y_train, no_segments)
            # y_test_segments = split_into_segments(y_test, no_segments)

            f_indexes_names = torch_rnn_lstm_gru(model_name, device, no_segments, X_train_segments, y_train_segments,
                                                 X_test, y_test, 0.1, 100,
                                                 feat, f_indexes_names, iteration)
        case 'GRU':
            # Splitting data to shorter arrays, to avoid GPU overclocking
            no_segments = 2
            X_train_segments = split_into_segments(X_train, no_segments)
            # X_test_segments = split_into_segments(X_test, no_segments)
            y_train_segments = split_into_segments(y_train, no_segments)
            # y_test_segments = split_into_segments(y_test, no_segments)

            f_indexes_names = torch_rnn_lstm_gru(model_name, device, no_segments, X_train_segments, y_train_segments,
                                                 X_test, y_test, 0.1, 100, feat,
                                                 f_indexes_names, iteration)
        # case 'SVM':
        #     # Support Vector Machine
        case 'RF':
            run_random_forest(feat, X_train, y_train, X_test, y_test)

    wandb.finish()

    return f_indexes_names


def split_into_segments(data, no_arrays):
    segment_length = len(data) // no_arrays
    data_splits = []
    for i in range(no_arrays):
        start_index = i * segment_length
        end_index = (i + 1) * segment_length
        data_split = data[start_index:end_index]
        data_splits.append(data_split)
    return data_splits


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
    model.fit(ppg_train, abp_train)
    predictions = model.predict(ppg_test)
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


def training_loop(model_name, model, criterion, optimizer, num_epochs, X_train, y_train):
    for epoch in range(num_epochs):
        # Forward pass and loss
        y_predicted = model(X_train)
        loss = criterion(y_predicted, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({f"{model_name} train loss (MSE)": loss})

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print('--------')


def testing_evaluating(model_name, model, X_test, y_test, learning_rate, num_epochs, feat, plot=False):
    with torch.no_grad():
        # Testing
        y_pred = model(X_test)

        # Converting back to numpy
        y_pred = y_pred.to('cpu').numpy()
        X_test, y_test = X_test.to('cpu').numpy(), y_test.to('cpu').numpy()

        # Evaluating and Logging
        mse, mae, r2, bias, loa_l, loa_u = evaluate(y_test, y_pred)
        rmse = np.sqrt(mse)
        logging.info(f'PyTorch {model_name}: learning_rate={learning_rate}, epochs={num_epochs} ({feat})\n'
                     f'\t\t\t\t\t  MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.3f}, R^2: {r2:.3f}, '
                     f'Bias: {bias:.3f}, LoA: ({loa_l:.3f}, {loa_u:.3f})')

        wandb.log({"test RMSE": rmse, "test MAE": mae})

        # Plotting
        if plot:
            visual.plot_ml_features_line(f'PyTorch {model_name}', y_test, y_pred)


def feature_importance_permutation(model, X_test, y_test, no_features, feature_indexes, feature_names):
    baseline_performance = calculate_performance(model, X_test, y_test)
    feature_importances = np.array([], dtype=np.float32)
    for j in range(no_features):
        # Copy original data
        X_permuted = X_test.clone()
        # Shuffle the values of the j-th feature
        X_permuted[:, j] = torch.rand(X_permuted.shape[0])
        # Calculate performance with permuted feature
        permuted_performance = calculate_performance(model, X_permuted, y_test)
        # Calculate feature importance as the decrease in performance
        feature_importances = np.append(feature_importances, -(baseline_performance - permuted_performance))
    feature_weights = [(index, name, importance)
                       for index, name, importance
                       in zip(feature_indexes, feature_names, feature_importances)]
    sorted_feature_weights = sorted(feature_weights, key=lambda x: x[2], reverse=True)
    return np.array(sorted_feature_weights)


def calculate_performance(model, X, y_true):
    with torch.no_grad():
        y_pred = model(X)
    return mean_squared_error(y_true.cpu(), y_pred.cpu())


def convert_to_tensors(device, X_train, y_train, X_test, y_test):
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)
    return X_train, y_train, X_test, y_test


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def torch_regression(model_name, device, X_train, y_train, X_test, y_test, learning_rate, num_epochs, feat):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    X_train, y_train, X_test, y_test = convert_to_tensors(device, X_train, y_train, X_test, y_test)

    # Instantiate the model, loss function, and optimizer
    model = LinearRegression(input_size=input_size, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training
    training_loop(model_name, model, criterion, optimizer, num_epochs, X_train, y_train)

    # Testing and Evaluating
    testing_evaluating(model_name, model, X_test, y_test, learning_rate, num_epochs, feat, plot=False)


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


def torch_neural_net(model_name, device, X_train, y_train, X_test, y_test, learning_rate, num_epochs, feat):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    hidden_size = int((input_size + output_size) / 2)

    X_train, y_train, X_test, y_test = convert_to_tensors(device, X_train, y_train, X_test, y_test)

    model = NeuralNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    training_loop(model_name, model, criterion, optimizer, num_epochs, X_train, y_train)

    # Testing and Evaluating
    testing_evaluating(model_name, model, X_test, y_test, learning_rate, num_epochs, feat, plot=False)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_features):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        h_t = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float32)
        h_t2 = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float32)
        c_t2 = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float32)
        h_t, c_t = self.lstm1(x, (h_t, c_t))
        h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        output = self.linear(h_t2)
        return output


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, out_features):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru1 = nn.GRUCell(input_size, hidden_size)
        self.gru2 = nn.GRUCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        h_t = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float32)
        h_t2 = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float32)
        h_t = self.gru1(x, h_t)
        h_t2 = self.gru2(h_t, h_t2)
        output = self.linear(h_t2)
        return output


def torch_rnn_lstm_gru(model_name, device, no_segments, X_train_segments, y_train_segments,
                       X_test, y_test, learning_rate, num_epochs, feat, feature_names, iteration):
    input_size = X_train_segments[0].shape[1]
    output_size = y_train_segments[0].shape[1]
    hidden_size = input_size * 2

    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    if model_name == 'GRU':
        model = GRU(input_size, hidden_size, output_size)
    else:
        model = LSTM(input_size, hidden_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)

    # Training loop
    best_loss = float('inf')
    patience, counter = 3, 0
    print(model_name, iteration)
    for epoch in range(num_epochs):
        train_loss = float('inf')
        for i in range(no_segments):
            X_train = torch.from_numpy(X_train_segments[i]).float().to(device)
            y_train = torch.from_numpy(y_train_segments[i]).float().to(device)

            def closure():
                optimizer.zero_grad()
                y_train_pred = model(X_train)
                loss = criterion(y_train_pred, y_train)
                loss.backward()
                return loss

            optimizer.step(closure)
            train_loss = closure().item()

        # Clearing Tensors
        X_train, y_train = None, None
        wandb.log({"training loss (MSE)": train_loss}, step=epoch)

        # Intermittent testing
        with torch.no_grad():
            y_test_pred = model(X_test)
            test_loss = criterion(y_test_pred, y_test)
            wandb.log({"testing loss (MSE)": test_loss}, step=epoch)

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Training Loss: {train_loss:.4f},'
                  f' Testing Loss: {test_loss:.4f}')

        # Check for early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1} as training loss starts increasing')
                break

    # Model Saving
    torch.save(model, f'models/{model_name}_{iteration}')

    # Testing and Evaluating
    testing_evaluating(model_name, model, X_test, y_test, learning_rate, num_epochs, feat, plot=False)

    # Feature Importance Permutation
    f_indexes = feature_names[:, 0]
    f_names = feature_names[:, 1]
    weights = feature_importance_permutation(model, X_test, y_test, X_train_segments[0].shape[1], f_indexes, f_names)
    visual.plot_feature_importances(weights, model_name, iteration)

    # Feature Reduction
    new_features = None
    if iteration == 0:
        new_features = weights[:25, :2]
    if iteration == 1:
        new_features = weights[:15, :2]
    if iteration == 2:
        new_features = weights[:7, :2]

    return new_features


def main():
    feature_names = pd.read_csv('./features/feature_indexes_names.csv', header=None).values.tolist()
    f_indexes_names = np.array(feature_names)
    for iteration in range(1):
        train_test_model('RF', f_indexes_names, iteration)
        # run_model('LR', f_indexes_names, iteration)
        # run_model('MLP', f_indexes_names, iteration)
        # f_indexes_names = run_model('LSTM', f_indexes_names, iteration)
        # f_indexes_names = run_model('GRU', f_indexes_names, iteration)

    # validate_model()


if __name__ == "__main__":
    main()
