import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize

def run_kfold(dataset, predictors, response, model, splits):
    predictor = normalize(dataset[predictors])
    responses = dataset[response]
    kscore = cross_val_score(model, predictor, responses, cv = splits).sum() / splits
    print("Kfold " + str(splits) + " score: ", kscore)

def main():

    games = pd.read_csv('games_new_vars.csv')

    predictor = ['abs_diff_rating']
    response = 'higher_rating_won'
    lg = LogisticRegression()

    FIRST_SPLIT = 5
    SECOND_SPLIT = 10

    print("K scores for single predictor logistic regression:")
    run_kfold(games, predictor, response, lg, FIRST_SPLIT)
    run_kfold(games, predictor, response, lg, SECOND_SPLIT)


    X_train, x_test, y_train, y_test = train_test_split(games.abs_diff_rating,
                                                        games.higher_rating_won,
                                                        test_size = 0.33, random_state = 1)
    lg.fit(X_train.values.reshape(-1,1), y_train.values.ravel())
    predictions = lg.predict(x_test.values.reshape(-1,1))
    report = classification_report(predictions, y_test)
    print("Classification report: ")
    print(report)

    print("K scores for multiple logistic regression:")
    predictors = ['abs_diff_rating', 'higher_rating_coded', 'turns', 'white_higher_rated']
    run_kfold(games, predictors, response, lg, FIRST_SPLIT)
    run_kfold(games, predictors, response, lg, SECOND_SPLIT)

    X_train, x_test, y_train, y_test = train_test_split(normalize(games[predictors]),
                                                        games.higher_rating_won,
                                                        test_size=0.33, random_state=1)
    lg.fit(X_train, y_train)
    predictions = lg.predict(x_test)
    report = classification_report(predictions, y_test)
    print("Classification report: ")
    print(report)


    print('Main completed')
    return 0

main()