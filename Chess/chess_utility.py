"""
Module containing utility functions for chess data set analysis.

"""

# General math
import numpy as np

# Model metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Visualization
import seaborn as sns

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# TODO test suites for functions

def sample_entries(data_set, sample_size, predictors):
    """ Sample rows from a data set.

    :param data_set: Data frame containing data.
    :param sample_size: Number of rows to sample.
    :param predictors: Predictors to get a sample for.
    :return: Sample data frame containing sampled rows.
    """
    random_indices = np.random.randint(len(data_set), size=sample_size)
    return data_set.iloc[random_indices][predictors]


def run_k_fold(data_set, X, y, model, splits):
    """ Run K fold cross validation.

    :param data_set: Data frame containing data.
    :param X: Predictor variables.
    :param y: Target variable.
    :param model: Model to fit data on.
    :param splits: K in K fold cross validation.
    :return: Accuracy of model
    """

    predictors = data_set[X]
    responses = data_set[y]
    k_score = cross_val_score(model, predictors, responses, cv = splits).sum() / splits
    return k_score


def classification_model_results(predictions, true_responses):
    """ Print and return a confusion matrix and its associated classification report.

    :param predictions: Model predictions.
    :param true_responses: True responses.
    :return: 2d array containing confusion matrix, string containing classification report.
    """
    print(confusion_matrix(predictions, true_responses))
    print(classification_report(predictions, true_responses))
    return confusion_matrix(predictions, true_responses), classification_report(predictions, true_responses)


def run_stratified_k(model, num_splits, X, y, random_state):
    """ Run a stratified K fold cross validation.

    :param model: Statistical learning model to use
    :param num_splits: K in K fold cross validation.
    :param X: Predictor variable.
    :param y: Target variable.
    :param random_state: Randomization seed.
    :return: 2d array containing classification results
    """

    skf = StratifiedKFold(n_splits = num_splits, shuffle = True, random_state = random_state)
    skf.get_n_splits(X, y)

    sum_confusion_matrix = 0

    for train_index, test_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        c_matrix = confusion_matrix(predictions, y_test)  # TODO Include multi label situation if necessary

        sum_confusion_matrix += c_matrix

    return sum_confusion_matrix


def create_cm_plot(title, confusion_matrix):
    """ Get a heat map of classification results for a binary confusion matrix.

    :param title: Title to give the plot.
    :param confusion_matrix: 2d array containing classification results
    :return: subplot of classification results
    """

    cm_plot = sns.heatmap(confusion_matrix,
                          annot = True,
                          xticklabels = ['Loss', 'Win'],
                          yticklabels = ['Loss', 'Win'],
                          fmt = 'd',
                          linewidths = 0.5,
                          cbar = False,
                          cmap = 'Blues')

    cm_plot.set_title(title)
    cm_plot.set_ylabel('True Values')
    cm_plot.set_xlabel('Predicted Values')

    return cm_plot


def get_accuracy(confusion_matrix):
    """ Calculate accuracy from a confusion matrix. In chess data set case, how correct were our
    predictions (both for wins and losses)?

    :param confusion_matrix: 2d array containing classification results
    :return: accuracy
    """
    return confusion_matrix.diagonal().sum() / confusion_matrix.sum()


def get_precision(confusion_matrix):
    """ Calculate precision of a confusion matrix. In chess data set case, how often do we predict the
    win?

    :param confusion_matrix: 2d array containing classification results
    :return: precision
    """
    return confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])


def get_recall(confusion_matrix):
    """ Calculate recall from a confusion matrix. In chess data set case, how well did we correctly predict
    the wins?

    :param confusion_matrix: 2d array containing classification results
    :return: recall
    """
    return confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])


def get_f_measure(confusion_matrix):
    """ This will take in a confusion matrix and return the f measure for the matrix.
    An f measure combines precision and recall into a single measurement.


    :param confusion_matrix: 2d array containing classification results
    :return: f_measure of the matrix
    """
    numerator = 2 * get_recall(confusion_matrix) * get_precision(confusion_matrix)
    denominator = get_recall(confusion_matrix) + get_precision(confusion_matrix)
    return numerator / denominator


def get_specificity(confusion_matrix):
    """ Returns the specificity of a confusion matrix. In chess data set terms, how many of the losses did
    we predict correctly?

    :param confusion_matrix: 2d array containing classification results
    :return: specificity
    """

    return confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])


def get_negative_pv(confusion_matrix):
    """ Acquires the negative predictive value binary classification case. In chess data set terms, how often
    do we predict a loss for the higher rated player?

    :param confusion_matrix: 2d array containing classification results
    :return: negative predicted value
    """

    return confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])


def get_cm_results(confusion_matrix):
    """ Takes in a confusion matrix and gives a dictionary containing useful confusion matrix measures.

    :param confusion_matrix: 2d array containing classification results
    :return: dictionary containing results
    """

    results = {'accuracy': get_accuracy(confusion_matrix), 'precision': get_precision(confusion_matrix),
               'recall': get_recall(confusion_matrix), 'fmeasure': get_f_measure(confusion_matrix),
               'specificity': get_specificity(confusion_matrix), 'negative_pv': get_negative_pv(confusion_matrix)}

    return results


def run_logistic_regression(number_folds, X, y, random_state):
    """ Fit a logistic regression on X and y. Run K fold cross validation and get a classification confusion matrix.

    :param number_folds: Number K in K fold cross validation.
    :param X: Predictor variables.
    :param y: Target variable.
    :param random_state: Randomization seed.
    :return: 2d array containing classification matrix.
    """
    lg = LogisticRegression()

    lg_confusion_matrix = run_stratified_k(lg, number_folds, X, y, random_state = random_state)

    return lg_confusion_matrix


def run_bag_trees(number_folds, number_estimators, random_state, X, y):
    """ Fit a bag of trees to X and y. Run K fold cross validation and return a classifcation confusion matrix.

    :param number_folds: Number K in K fold cross validation.
    :param number_estimators: Number of estimators to use in the bag.
    :param random_state: Randomization seed.
    :param X: Predictor variable.
    :param y: Target variable.
    :return: 2d array classification confusion matrix
    """

    tree = DecisionTreeClassifier()

    bag = BaggingClassifier(base_estimator = tree,
                            n_estimators = number_estimators,
                            random_state = random_state)

    bag.fit(X, y)

    bag_confusion_matrix = run_stratified_k(bag, number_folds, X, y, random_state)

    return bag_confusion_matrix


def run_random_forest(number_folds, number_estimators, random_state, X, y):
    """ Fit a random forest to X and y. Get a confusion matrix with classification results.

    :param number_folds: Number K in K fold cross validation.
    :param number_estimators: Number of trees to use in the forest.
    :param random_state: Randomization seed.
    :param X: Predictor variable.
    :param y: Target variable
    :return: 2d array containing a classification confusion matrix
    """

    random_forest = RandomForestClassifier(n_estimators = number_estimators, random_state = random_state)

    r_forest_confusion_matrix = run_stratified_k(random_forest, number_folds, X, y, random_state)

    return r_forest_confusion_matrix
