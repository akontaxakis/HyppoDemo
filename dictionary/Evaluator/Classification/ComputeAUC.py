from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score


class ComputeAUC(BaseEstimator, TransformerMixin):
    def __init__(self, y_true=None):
        self.y_true = y_true

    def fit(self, X, y=None):
        # Assuming y contains the true labels during fit
        self.y_true = X
        return self

    def score(self, X):
        # Assuming X contains the probabilities during score
        auc = roc_auc_score(self.y_true, X)
        return auc

