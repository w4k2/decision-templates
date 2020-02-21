import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class VotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.classifiers_pool_ = self.estimators

        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['classes_'])
        X = check_array(X)

        return np.array([
            np.array(list(map(lambda x: x == self.classes_, est.predict(X))))
            for est in self.classifiers_pool_
        ]).sum(axis=0)

    def predict(self, X):
        return self.classes_.take(np.argmax(self.predict_proba(X), axis=1))
