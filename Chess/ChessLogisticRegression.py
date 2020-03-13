# Create a logistic regression model

import pandas as pd
from pandas import DataFrame, Series
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# TODO the score outputed here may not be the error rate
def kfold_logistic_regression(dataset, predictors, response, splits):

    lg = LogisticRegression()

    # Perform k-fold cross validation for k = splits
    kf = KFold(n_splits=splits)

    # We will find the estimated error rate
    estimated_error_rate = 0

    for train, test in kf.split(dataset):

        # Create training and test data
        train_data = dataset.iloc[train][predictors + response]
        test_data = dataset.iloc[test][predictors + response]

        # Predictor and response (training)
        X_train = DataFrame(train_data[predictors])
        Y_train = DataFrame(train_data[response])

        # Fit the model
        lg.fit(X_train, Y_train)

        # Predictor and response(test)
        X_test = DataFrame(test_data[predictors])
        Y_test = DataFrame(test_data[response])

        # Store the score for the model
        estimated_error_rate += lg.score(X_test, Y_test)

    estimated_error_rate = estimated_error_rate / splits
    return estimated_error_rate, lg

def main():

    games = pd.read_csv('games_new_vars.csv')

    predictors = ['abs_diff_rating']
    response = ['higher_rating_won']

    # Run kfold cv for 5 splits
    FIRST_SPLIT = 5
    kfive_error_rate, kfive_model = kfold_logistic_regression(games, predictors, response, FIRST_SPLIT)
    print("Kfive estimated error rate:", kfive_error_rate)

    # Run kfold cv 10 splits
    SECOND_SPLIT = 10
    kten_error_rate, kten_model = kfold_logistic_regression(games, predictors, response, SECOND_SPLIT)
    print("Kten estimated error rate:", kten_error_rate)

    print('Main completed')
    return 0

main()