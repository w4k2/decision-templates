import numpy as np

from .decision_templates_base import make_decision_profiles, make_decision_templates

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.cluster import OPTICS


class OpticsTemplatesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, use_weights=False):
        self.estimators = estimators
        self.use_weights = use_weights

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        self.classifiers_pool_ = self.estimators

        dp = make_decision_profiles(X, self.classifiers_pool_)

        self.decision_templates_ = []
        self.decision_templates_classes_ = []
        self.decision_templates_weights_ = []

        for label in np.unique(y):
            cl = OPTICS(min_samples=3).fit(dp[y == label])

            dt, _ = make_decision_templates(dp[y == label], cl.labels_)
            self.decision_templates_.append(dt)
            self.decision_templates_classes_.append(np.repeat(label, len(dt)))

            if self.use_weights:
                labels, counts = np.unique(cl.labels_, return_counts=True)
                self.decision_templates_weights_.append(counts / counts.sum())
                # TODO: Consider variance as alternative
            else:
                self.decision_templates_weights_.append(np.ones(len(dt)))

        self.decision_templates_ = np.concatenate(self.decision_templates_)
        self.decision_templates_classes_ = np.concatenate(self.decision_templates_classes_)
        self.decision_templates_weights_ = np.concatenate(self.decision_templates_weights_)

        return self

    def predict(self, X):
        check_is_fitted(self, ['classes_', 'classifiers_pool_', 'decision_templates_', 'decision_templates_classes_', 'decision_templates_weights_'])
        X = check_array(X)

        dp = make_decision_profiles(X, self.classifiers_pool_)
        distances = np.array([np.linalg.norm(x - dp, axis=1) / w for x, w in zip(self.decision_templates_, self.decision_templates_weights_)])

        return self.decision_templates_classes_.take(np.argmin(distances, axis=0))
