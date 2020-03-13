import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def run_kfold(dataset, predictors, response, model, splits):
    predictor = dataset[predictors]
    responses = dataset[response]
    kscore = cross_val_score(model, predictor, responses, cv=splits).sum() / splits
    print("Kfold " + str(splits) + " score: ", kscore)

def main():

    games = pd.read_csv('games_new_vars.csv')

    predictor = ['abs_diff_rating']
    response = 'higher_rating_won'
    lg = LogisticRegression()

    print("Scores for single predictor logistic regression:")
    FIRST_SPLIT = 5
    run_kfold(games, predictor, response, lg, FIRST_SPLIT)

    SECOND_SPLIT = 10
    run_kfold(games, predictor, response, lg, SECOND_SPLIT)

    print("Scores for multiple logistic regression:")
    predictors = ['abs_diff_rating', 'higher_rating_coded', 'turns', 'white_higher_rated']
    run_kfold(games, predictors, response, lg, FIRST_SPLIT)
    run_kfold(games, predictors, response, lg, SECOND_SPLIT)

    print('Main completed')
    return 0

main()