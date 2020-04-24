"""
Module containing utility functions for chess data set analysis.

"""

# General math
import numpy as np

# Data frame manipulation
import pandas as pd

# Model metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

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
        c_matrix = confusion_matrix(predictions, y_test)

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

    return confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])


def get_negative_pv(confusion_matrix):
    """ Acquires the negative predictive value binary classification case. In chess data set terms, how often
    do we predict a loss for the higher rated player?

    :param confusion_matrix: 2d array containing classification results
    :return: negative predicted value
    """

    return confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])


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


def create_bar_results(model_results, title, ax):
    """ Create a seaborn bar plot of a dictionary containing model results.

    :param model_results: A dictionary containing the model's confusion matrix readings
    :param title: Title of the subplot
    :param ax: The axis of the subplot, should be passed in as an entry from plt.subplots()
    :return: None
    """
    series_results = pd.Series(model_results)

    sns.barplot(series_results.index, series_results.values, ax = ax)
    ax.title.set_text(title)

    ax = ax

    for bar, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(
            bar,
            height + 0.02,
            u'{:0.3f}'.format(height),
            color = 'black',
            fontsize = 14,
            ha = 'center',
            va = 'center'
        )

    return None


def create_cumulative_results_plot(r_forest_results, lg_results, bag_results):
    """ Create a grid of subplots containing bar plots for model results.

    :param r_forest_results: Dictionary containing results from a random forest.
    :param lg_results: Dictionary containing results from a logistic regression.
    :param bag_results: Dictionary containing results from a bag of trees.
    :return: None
    """
    fig, axs = plt.subplots(figsize=[20, 5], ncols=3, sharey=True, sharex=True, gridspec_kw={'wspace': 0.2})

    create_bar_results(lg_results, 'Logistic Regression Results', axs[0])
    create_bar_results(bag_results, 'Bag Results', axs[1])
    create_bar_results(r_forest_results, 'Random Forest Results', axs[2])

    plt.ylim([0, 1])

    return None

def group_results(measure, lg_results, bag_results, r_forest_results):
    """ Construct and return a dictionary containing the same measure across different models.

    :param measure: Type of measure to get from the results e.g. accuracy.
    :param lg_results: Logistic regression results.
    :param bag_results: Bag of trees results.
    :param r_forest_results: Random forest results.
    :return: Dictionary containing measures for each model.
    """

    results = {'Logistic Regression': lg_results[measure], 'Bag of Trees':bag_results[measure], 'Random Forest':r_forest_results[measure]}
    return results

def group_important_results(lg_results, bag_results, r_forest_results):
    """ Create and return dictionaries with results across models.

    :param lg_results: Logistic regression results.
    :param bag_results: Bag of trees results.
    :param r_forest_results: Random forest results.
    :return: Six dictionaries containing results across models.
    """

    accuracy = group_results('accuracy', lg_results, bag_results, r_forest_results)
    precision = group_results('precision', lg_results, bag_results, r_forest_results)
    recall = group_results('recall', lg_results, bag_results, r_forest_results)
    fmeasure = group_results('fmeasure', lg_results, bag_results, r_forest_results)
    specificity = group_results('specificity', lg_results, bag_results, r_forest_results)
    negative_pv = group_results('negative_pv', lg_results, bag_results, r_forest_results)

    return accuracy, precision, recall, fmeasure, specificity, negative_pv

def create_specific_results_plot(r_forest_results, lg_results, bag_results, ncols, nrows):
    """ Create a grid of subplots containing bar plots for model results.

    :param r_forest_results: Dictionary containing results from a random forest.
    :param lg_results: Dictionary containing results from a logistic regression.
    :param bag_results: Dictionary containing results from a bag of trees.
    :return: None
    """
    fig, axs = plt.subplots(figsize=[20, 5], ncols = ncols, nrows = nrows ,sharey=True, sharex=True, gridspec_kw={'wspace': 0.2})

    accuracy, precision, recall, fmeasure, specificity, negative_pv = group_important_results(lg_results,
                                                                                                 bag_results,
                                                                                                 r_forest_results)

    create_bar_results(accuracy, 'Accuracy', axs[0][0])
    create_bar_results(precision, 'Precision', axs[0][1])
    create_bar_results(recall, 'Recall', axs[0][2])

    create_bar_results(fmeasure, 'F-measure', axs[1][0])
    create_bar_results(specificity, 'Specificity', axs[1][1])
    create_bar_results(negative_pv, 'Negative PV', axs[1][2])

    plt.ylim([0, 1])

    return None

def get_recall_three_class(three_class_confusion_matrix, label):
    """ Find the recall of a label in a three class confusion matrix

    :param three_class_confusion_matrix: Confusion matrix with three labels, 2d array
    :param label: Either 0, 1, or 2 to denote loss, draw, or win respectively.
    :return: Recall metric for the label indicated.
    """

    recall = three_class_confusion_matrix[ label][ label] / (three_class_confusion_matrix[ label][ 0] +
                                                             three_class_confusion_matrix[ label][ 1] +
                                                             three_class_confusion_matrix[ label][ 2])
    return recall

def get_precision_three_class(three_class_confusion_matrix, label):
    """ Find the precision of a label in a three class confusion matrix.

    :param three_class_confusion_matrix: Confusion matrix with three labels, 2d array
    :param label: Either 0, 1, or 2 to denote loss, draw, or win respectively.
    :return: Precision metric for the label indicated.
    """
    precision = three_class_confusion_matrix[label][label] / (three_class_confusion_matrix[0][label] +
                                                              three_class_confusion_matrix[1][label] +
                                                              three_class_confusion_matrix[2][label])
    return precision

def get_accuracy_three_class(confusion_matrix):
    num = confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2]
    den = confusion_matrix.sum()

    return num / den


def precision_recall_three_class(confusion_matrix, label):
    """ Get precision and recall for a label in a three class confusion matrix.

    :param confusion_matrix: Confusion matrix with three labels, 2d array
    :param label: Either 0, 1, or 2 to denote loss, draw, or win respectively.
    :return: Dictionary containing
    """
    precision = get_precision_three_class(confusion_matrix, label)
    recall = get_recall_three_class(confusion_matrix, label)
    results = {'Precision': precision, 'Recall': recall}
    return results

def make_multi_precision_recall_list(name_one, confusion_matrix_one, name_two, confusion_matrix_two, label):
    """ Take names and confusion matrices to create precision and recall dictionary.

    :param name_one: String naming model used in confusion matrix no 1
    :param confusion_matrix_one: Three class confusion matrix
    :param name_two: String naming model used in confusion matrix no 2
    :param confusion_matrix_two: Three class confusion matrix
    :param label: Either 0, 1, or 2 to denote loss, draw, or win respectively.
    :return: Dictionary containing precision and recall results.
    """
    results = { name_one: precision_recall_three_class( confusion_matrix_one, label ),
              name_two:  precision_recall_three_class( confusion_matrix_two, label )}
    return results


def create_specific_results_plot_general(results, ncols, nrows):
    """ Create subplots for results. More general than create_specific_results_plot.

    :param results: Dictionary of dictionaries containing data to plot.
    :param ncols: Columns in figure for subplots.
    :param nrows: Rows in figure for subplots.
    :return: None
    """
    fig, axs = plt.subplots(figsize=[20, 5],
                            ncols=ncols,
                            nrows=nrows,
                            sharey=True,
                            sharex=True,
                            gridspec_kw={'wspace': 0.2})

    col = 0
    for key in results:
        create_bar_results(results[key], key, axs[col])
        col += 1

    plt.ylim([0, 1])

    return None

def make_plot_multi_label( bag_multi_confusion_matrix, r_forest_multi_confusion_matrix, label ):
    """ Create a plot comparing a bag of trees and random forest precision and recall of a label.

    :param bag_multi_confusion_matrix: Confusion matrix for a bag of trees. Three classes.
    :param r_forest_multi_confusion_matrix: Confusion matrix for a random forest. Three classes.
    :param label: Either 0, 1, or 2 to denote loss, draw, or win respectively.
    :return: None
    """
    results = make_multi_precision_recall_list( 'Bag of trees', bag_multi_confusion_matrix,
                                                  'Random forest', r_forest_multi_confusion_matrix, label )
    create_specific_results_plot_general( results,  2, 1)