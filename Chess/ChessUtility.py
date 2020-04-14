# Utility pack for chess project -> and maybe later, future projects.
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns

# TODO test suites for functions

def sample_entries(dataset, sample_size, predictors):
    """Pass in predictors as a list of strings. """
    random_indices = np.random.randint(len(dataset), size=sample_size)
    return dataset.iloc[random_indices][predictors]

def run_kfold(dataset, predictor_variables, response_variables, model, splits):
    predictors= dataset[predictor_variables]
    responses = dataset[response_variables]
    kscore = cross_val_score(model, predictors, responses, cv = splits).sum() / splits
    print(kscore)
    return kscore

def classification_model_results(predictions, true_responses):
    print(confusion_matrix(predictions, true_responses))
    print(classification_report(predictions, true_responses))
    return confusion_matrix(predictions, true_responses), classification_report(predictions, true_responses)


def run_stratified_k(model, num_splits, X, y, random_state):
    """ Take a model along with predictors and responses to run a stratified k fold cv.

    model - predictive model to use, num_splits - number of folds, y - response variable, X - predictor variable/s
    list containing average accuracy over folds, sum of each confusion matrix
    """

    skf = StratifiedKFold(n_splits=num_splits, shuffle = True,  random_state = random_state)
    skf.get_n_splits(X, y)

    sum_confusion_matrix = 0

    for train_index, test_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        c_matrix = confusion_matrix(predictions, y_test)    # TODO Include multi label situation if necessary

        sum_confusion_matrix += c_matrix

    return sum_confusion_matrix

# Test suite

def create_cm_plot(title, confusion_matrix):
    """ Creates a plot for a binary classifier's confusion matrix.

    title -> matplotlib sub plot
    """

    cm_plot = sns.heatmap(confusion_matrix,
                          annot=True,
                          xticklabels=['Loss', 'Win'],
                          yticklabels=['Loss', 'Win'],
                          fmt='d',
                          linewidths=0.5,
                          cbar=False,
                          cmap='Blues')

    cm_plot.set_title(title)
    cm_plot.set_ylabel('True Values')
    cm_plot.set_xlabel('Predicted Values')

    return cm_plot


def get_accuracy(confusion_matrix):
    """ Takes a confusion matrix and gives the accuracy of the classifier.

    confusion matrix  2d array -> accuracy of predictions
    """
    return confusion_matrix.diagonal().sum() / confusion_matrix.sum()


def get_precision(confusion_matrix):
    """ Rate of positively predicting a win for higher rated players.

    confusion_matrix 2d array -> precision
    """
    return confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])


def get_recall(confusion_matrix):
    """ Out of all the true positive classes (in this case wins), how many did we predict correctly.

    confusion matrix 2d array -> recall
    """
    return confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])

def get_fmeasure(confusion_matrix):
    """ This will take in a confusion matrix and return the f measure for the matrix.


    :param confusion_matrix:
    :return: f_measure of the matrix
    """
    numerator = 2 * get_recall(confusion_matrix) * get_precision(confusion_matrix)
    denominator = get_recall(confusion_matrix) + get_precision(confusion_matrix)
    return numerator / denominator

def get_specificity(confusion_matrix):
    """ Returns the specificity of the confusion matrix, in our case getting right amount of losses.

    :param confusion_matrix
    :return: specificity
    """

    return confusion_matrix[0][0] / ( confusion_matrix[0][0] + confusion_matrix[0][1] )

def get_negative_pv(confusion_matrix):
    """ Acquires the negative predictive value, in the chess binary classification case, how often
    do we predict a loss for the higher rated player.

    :param confusion_matrix:
    :return: negative predicted value
    """

    return confusion_matrix[0][0] / ( confusion_matrix[0][0] + confusion_matrix[1][0] )