# Make a simple decision tree

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

def main():

    games = pd.read_csv("games_new_vars.csv")

    X = games[['abs_diff_rating', 'white_higher_rated', 'turns']]
    Y = games['result']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

    tree = DecisionTreeClassifier().fit(X_train, y_train)
    ypreds = tree.predict(X_test)

    c_matrix = metrics.confusion_matrix(ypreds, y_test)
    print(c_matrix)

    report = metrics.classification_report(ypreds, y_test)
    print(report)


    print("Main completed.")
    return 0

main()