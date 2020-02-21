import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


RULE_DICT = {
    'max': np.max,
    'min': np.min,
    'avg': np.mean,
    'prd': np.prod
}


class AggregationEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, rule):
        self.estimators = estimators
        self.rule = rule

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)
        self.classifiers_pool_ = self.estimators

        assert self.rule in RULE_DICT

        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['classes_'])
        X = check_array(X)

        predictions = []

        for est in self.classifiers_pool_:
            predictions.append(est.predict_proba(X))

        predictions = np.array(predictions)

        return RULE_DICT[self.rule](predictions, axis=0)

    def predict(self, X):
        return self.classes_.take(np.argmax(self.predict_proba(X), axis=1))
