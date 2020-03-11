# Module for creating variables needed for personal chess data set analysis

import pandas as pd
import numpy as np

def make_higher_rating_coded(games):
    # Next, we will code higher rating as a binary so that we can easily use it
    games['higher_rating_coded'] = 0
    games.loc[games.higher_rating == 'white', 'higher_rating_coded'] = 1

def make_higher_rating_won(games):
    # Now, we encode a variable that returns 1 if the higher rating won and 0 otherwise.
    games['higher_rating_won'] = 0
    games.loc[games.winner == games.higher_rating, 'higher_rating_won'] = 1

def make_higher_rating(games):
    # Create a new variable to determine who has the higher rating
    games['higher_rating'] = ''
    # Given the organization of the data, positive differences in the diff_rating indicate white had a higher rating,
    # whereas negative ones indicated a higher rating for black. A difference of zero, of course, indicates an equal rating.
    games.loc[games.diff_rating > 0, 'higher_rating'] = 'white'
    games.loc[games.diff_rating < 0, 'higher_rating'] = 'black'
    games.loc[games.diff_rating == 0, 'higher_rating'] = 'same'

def make_abs_difference(games):
    # Create a new variable with absolute value of differences
    games['abs_diff_rating'] = np.abs(games['diff_rating'])

def make_difference_rating(games):
    # Create a new variable for the difference in rating between two players.
    games['diff_rating'] = games.white_rating - games.black_rating

def main():

    games = pd.read_csv('games.csv')

    # Create variables for analysis
    make_difference_rating(games)
    make_abs_difference(games)
    make_higher_rating(games)
    make_higher_rating_won(games)
    make_higher_rating_coded(games)
    make_higher_rating_coded(games)

    # Print out a random sample of variables we created
    rand_ind = np.random.randint(len(games), size=10)
    print(games.iloc[rand_ind][['higher_rating','higher_rating_won','winner']].head(5))
    print(games.iloc[rand_ind][['higher_rating_coded', 'diff_rating','abs_diff_rating']].head(5))

    games.to_csv('games_new_vars.csv')

    print("Main completed.")
    return 0

main()