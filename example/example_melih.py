import os
import json
from functools import partial

from operator import itemgetter

import numpy as np

import sklearn
import sklearn.cross_validation
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


# Utility function to report best scores
# from http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html#example-model-selection-randomized-search-py
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def roc_auc_score(y_truth, y_pred, num_classes=None):
    def to_matrix(a, num_classes=None):
        if num_classes is None:
            # infer
            num_classes = np.max(a) - np.min(a) + 1
        r = np.zeros(shape=(a.shape[0], num_classes))
        r[np.arange(a.shape[0]), a] = 1
        return r
    return sklearn.metrics.roc_auc_score(*map(partial(to_matrix, num_classes=num_classes), [y_truth, y_pred]))


def run_dataset(dataset_name, X, y, seed=1):
    num_classes = np.unique(y).shape[0]
    X_train, X_test, y_train, y_test = \
        sklearn.cross_validation.train_test_split(X, y, random_state=seed)

    automl = autosklearn.classification.AutoSklearnClassifier(
        # important parameters
        ensemble_size=1,
        resampling_strategy='holdout',
        include_preprocessors=['polynomial', 'pca'],
        include_estimators=['lda',
                             'xgradient_boosting',
                             'qda',
                             'extra_trees',
                             'decision_tree',
                             'gradient_boosting',
                             'k_nearest_neighbors',
                             'multinomial_nb',
                             'libsvm_svc',
                             'gaussian_nb',
                             'random_forest',
                             'bernoulli_nb'],

        per_run_time_limit=30,
        ml_memory_limit=1000*16,

        # default parameters
        time_left_for_this_task=120,
        tmp_folder='/tmp/autoslearn_holdout_example_tmp',
        output_folder='/tmp/autosklearn_holdout_example_out',
    )

    # set metric to auc to match our performance metric
    automl.fit(X_train, y_train, dataset_name=dataset_name, metric="auc_metric")

    # Print the best models together with their scores - if all scores are
    # unreasonably bad (around 0.0) you should have a look into the logging
    # file to figure out the error
    # report(automl.grid_scores_)

    # Print the final ensemble constructed by auto-sklearn.
    # print(automl.show_models())

    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    # print(automl.sprint_statistics())

    predictions = automl.predict(X_test)
    auc = roc_auc_score(y_test, predictions, num_classes=num_classes)
    print("auc score", auc)
    return auc


def run_openml_dataset(did, seed=1):
    from openml import datasets
    try:
        dataset = datasets.get_dataset(did)
        X, y, categorical = dataset.get_data(target=dataset.default_target_attribute,
                                             return_categorical_indicator=True)
        return run_dataset(str(did), X, y, seed=1)
    except:
        return np.nan
    

def test_dataset():
    dataset_name="digits"
    digits = sklearn.datasets.load_digits()
    X = digits.data
    y = digits.target
    return run_dataset("digits", X, y, seed=1)
    
def main():
    dataset_ids = [1154, 1412, 773, 812, 906, 1164, 770, 1038, 911, 912, 913, 23, 1071, 285, 30, 799, 40475, 932, 4135, 732, 683, 1452, 943, 48, 821, 1471, 1600, 795, 908, 859, 971, 1100, 976, 722, 36, 1115, 1500, 862, 1488, 995, 741, 1126, 873, 679, 752, 1535, 1020, 894, 1151]
    # dataset_ids = dataset_ids[0:1]
    aucs = map(run_openml_dataset, map(str, dataset_ids))
    
    results = zip(dataset_ids, aucs)
    print results
    file_path = os.path.expanduser("~/scratch/auto-sklearn-melih-results/results.json")
    with open(file_path, "w") as fh:
        json.dump(results, fh)
    
if __name__ == '__main__':
    main()
