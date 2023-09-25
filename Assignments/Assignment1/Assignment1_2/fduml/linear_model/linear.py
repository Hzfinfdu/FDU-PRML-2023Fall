"""
Linear Model
"""

from abc import abstractmethod

class LinearModel(object):
    """Base class for Linear Models"""

    @abstractmethod
    def fit(self, X, y):
        """
        Fit model.

        Parameters
        ----------
        X : {array-like matrix} of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_targets)
            Target values.
        """

    @abstractmethod
    def predict(self, X):
        """
        Predict Y given X.

        Parameters
        ----------
        X : array-like matrix, shape (n_samples, n_features)
            Samples.
        """