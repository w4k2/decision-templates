import numpy as np


def make_decision_profiles(X, classifiers_pool):
    return np.concatenate(np.array([clf.predict_proba(X) for clf in classifiers_pool]), axis=1)


def make_decision_templates(decision_profiles, y):
    labels = np.unique(y)
    decision_templates = np.array([decision_profiles[y == _].mean(axis=0) for _ in labels])
    return decision_templates, labels
