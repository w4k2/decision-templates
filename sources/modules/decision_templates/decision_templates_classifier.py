import numpy as np

from .decision_templates_base import make_decision_profiles, make_decision_templates

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class DecisionTemplatesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        self.classifiers_pool_ = self.estimators

        dp = make_decision_profiles(X, self.classifiers_pool_)
        self.decision_templates_, self.decision_templates_classes_ = make_decision_templates(dp, y)

        return self

    def predict(self, X):
        check_is_fitted(self, ['classes_', 'classifiers_pool_', 'decision_templates_', 'decision_templates_classes_'])
        X = check_array(X)

        dp = make_decision_profiles(X, self.classifiers_pool_)
        distances = np.array([np.linalg.norm(x - dp, axis=1) for x in self.decision_templates_])

        return self.decision_templates_classes_.take(np.argmin(distances, axis=0))
