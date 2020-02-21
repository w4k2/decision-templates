import numpy as np
import pandas as pd
import os

from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from decision_templates import DecisionTemplatesClassifier, OpticsTemplatesClassifier
from ensembles import AggregationEnsemble, VotingEnsemble
from keel import load_dataset, find_datasets
from metrics import confusion_matrix_scores

DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets')
RESULTS_CSV = os.path.join(os.getcwd(), 'results.csv')

CLASSIFIERS = [
    ("Majority Voting", VotingEnsemble(None)),
    ("Avg Aggregation", AggregationEnsemble(None, rule='avg')),
    ("Decision Templates", DecisionTemplatesClassifier(None)),
    ("OPTICS Decision Templates", OpticsTemplatesClassifier(None)),
    ("Weighted OPTICS Decision Templates", OpticsTemplatesClassifier(None, use_weights=True)),
]


def prepare_pool(X, y):
    bbc = BalancedBaggingClassifier(base_estimator=GaussianNB(), n_estimators=20, random_state=0)
    bbc.fit(X, y)
    return bbc.estimators_

def main():
    print("Starting experiments")
    results = []

    for dataset_name in find_datasets(DATASETS_DIR):
        X, y = load_dataset(dataset_name, return_X_y=True, storage=DATASETS_DIR)
        print(f"Dataset: {dataset_name}")

        X = StandardScaler().fit_transform(X, y)

        folding = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=0)

        for fold_idx, (train_index, test_index) in enumerate(folding.split(X, y), 1):
            print(f"  Fold: {fold_idx}")

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            pool = prepare_pool(X_train, y_train)

            for clf_name, clf in CLASSIFIERS:
                clf = clone(clf).set_params(estimators=pool)
                print(f"    Classifier: {clf_name} ...")

                clf.fit(X_train, y_train)
                cm = confusion_matrix(y_test, clf.predict(X_test))

                results.append({
                    "Dataset": dataset_name,
                    "Classifier": clf_name,
                    "Fold": fold_idx,
                    **confusion_matrix_scores(cm)
                })

    # Store results to unprocessed csv
    pd.DataFrame(results).to_csv(RESULTS_CSV)


if __name__ == "__main__":
    main()
