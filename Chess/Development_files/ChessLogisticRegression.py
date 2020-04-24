# Create a logistic regression model

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

def kfold_logistic_regression(dataset, predictors, response, splits):

    lg = LogisticRegression()

    # Perform k-fold cross validation for k = splits
    kf = KFold(n_splits=splits)

    # We will find the estimated error rate
    estimated_accuracy = 0

    for train, test in kf.split(dataset):

        # Create training and test data
        train_data = dataset.iloc[train][predictors + response]
        test_data = dataset.iloc[test][predictors + response]

        # Predictor and response (training)
        X_train = train_data[predictors]
        Y_train = train_data[response].values.ravel()

        # Fit the model
        lg.fit(X_train, Y_train)

        # Predictor and response(test)
        X_test = test_data[predictors]
        Y_test = test_data[response]

        # Store the score for the model
        estimated_accuracy += lg.score(X_test, Y_test)

    estimated_accuracy = estimated_accuracy / splits
    return estimated_accuracy, lg

def main():

    games = pd.read_csv('games_new_vars.csv')

    predictors = ['abs_diff_rating']
    response = ['higher_rating_won']

    # Run kfold cv for 5 splits
    FIRST_SPLIT = 5
    kfive_accuracy, kfive_model = kfold_logistic_regression(games, predictors, response, FIRST_SPLIT)
    print("K five estimated accuracy:", kfive_accuracy)

    # Run kfold cv 10 splits
    SECOND_SPLIT = 10
    kten_accuracy, kten_model = kfold_logistic_regression(games, predictors, response, SECOND_SPLIT)
    print("K ten estimated accuracy:", kten_accuracy)

    print('Main completed')
    return 0

main()