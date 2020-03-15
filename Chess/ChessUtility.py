# Utility pack for chess project -> and maybe later, future projects.
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


def sample_entries(dataset, sample_size, predictors):
    """Pass in predictors as a list of strings. """
    random_indices = np.random.randint(len(dataset), size=sample_size)
    print(dataset.iloc[random_indices][predictors])
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
