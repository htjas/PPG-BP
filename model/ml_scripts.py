import sklearn


def create_model():
    print()


def main():
    print('ML scripts')
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
