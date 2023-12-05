import pandas as pd
import sklearn
import os

import sklearn.linear_model


def run_model():
    abs_path = os.path.abspath(os.getcwd())
    ppg_train_path = abs_path + '/features/training/total_median_cts_train.csv'
    df = pd.read_csv(ppg_train_path)
    values = df.values
    ppg_train = values[:, 0]

    ppg_test_path = abs_path + '/features/testing/total_median_cts_test.csv'
    df = pd.read_csv(ppg_test_path)
    values = df.values
    ppg_test = values[:, 0]

    # abp_train_path = abs_path + '/features/training/total_median_dia_train.csv'
    abp_train_path = abs_path + '/features/training/total_median_systoles_train.csv'
    df = pd.read_csv(abp_train_path)
    values = df.values
    abp_train = values[:, 0]

    abp_test_path = abs_path + '/features/testing/total_median_systoles_test.csv'
    df = pd.read_csv(abp_test_path)
    values = df.values
    abp_test = values[:, 0]

    # Assuming X contains CTs and y contains Systoles (or Diastoles)
    ppg_train = ppg_train[:len(abp_train)]
    ppg_train = ppg_train.reshape(-1, 1)
    abp_train = abp_train.reshape(-1, 1)

    model = sklearn.linear_model.LinearRegression()
    model.fit(ppg_train, abp_train)

    ppg_test = ppg_test[:len(abp_test)]
    ppg_test = ppg_test.reshape(-1, 1)

    # Make predictions on the test set
    predictions = model.predict(ppg_test)

    abp_test = abp_test.reshape(-1, 1)
    # Evaluate the model
    mse = sklearn.metrics.mean_squared_error(abp_test, predictions)
    print(f'Mean Squared Error: {mse}')


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
