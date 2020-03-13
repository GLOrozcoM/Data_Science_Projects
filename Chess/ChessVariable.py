# Module for creating variables needed for personal chess data set analysis

import pandas as pd
import numpy as np

def make_higher_rating_coded(games):
    games['higher_rating_coded'] = 0
    games.loc[games.higher_rating == 'white', 'higher_rating_coded'] = 1

def make_higher_rating_won(games):
    games['higher_rating_won'] = 0
    games.loc[games.winner == games.higher_rating, 'higher_rating_won'] = 1

def make_higher_rating(games):
    games['higher_rating'] = ''
    games.loc[games.diff_rating > 0, 'higher_rating'] = 'white'
    games.loc[games.diff_rating < 0, 'higher_rating'] = 'black'
    games.loc[games.diff_rating == 0, 'higher_rating'] = 'same'

def make_abs_difference(games):
    games['abs_diff_rating'] = np.abs(games['diff_rating'])

def make_difference_rating(games):
    games['diff_rating'] = games.white_rating - games.black_rating

def make_white_higher_rating(games):
    games['white_higher_rated'] = 0
    games.loc[games.higher_rating == 'white', 'white_higher_rated'] = 1

def make_result(games):
    games['result'] = 0
    games.loc[games.winner == 'draw', 'result'] = 1
    games.loc[games.higher_rating_won == 1, 'result'] = 2

def main():

    games = pd.read_csv('games.csv')

    make_difference_rating(games)
    make_abs_difference(games)
    make_higher_rating(games)
    make_higher_rating_won(games)
    make_higher_rating_coded(games)
    make_higher_rating_coded(games)
    make_white_higher_rating(games)
    make_result(games)

    random_indices = np.random.randint(len(games), size=10)
    print(games.iloc[random_indices][['higher_rating','higher_rating_won','winner']].head(5))
    print(games.iloc[random_indices][['higher_rating_coded', 'diff_rating','abs_diff_rating', 'result']].head(5))

    games.to_csv('games_new_vars.csv')

    print("Main completed.")
    return 0

main()