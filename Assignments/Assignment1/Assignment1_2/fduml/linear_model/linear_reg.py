"""
Linear Regression
"""
import numpy as np
from .linear import LinearModel

class LinearRegression(LinearModel):
    """
    Ordinary least squares Linear Regression.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Attributes
    ----------
    coef_ : array of shape (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features).

    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model.

    reg_ : (float)
        L2 regularization strength.
    """

    def __init__(self, reg=0.0):
        self.coef_ = None
        self.intercept_ = None

        self.reg_ = reg

    def fit(self, X, y):
        """
        Fit linear model.

        Parameters
        ----------
        X : {array-like matrix} of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            Fitted model with predicted self.coef_ and self.intercept_.
        """
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted coef_ and intercept_         #
        # in the self.coef_. and self.intercept_ respectively.                    #
        #                                                                         #
        # Notice:                                                                 #
        # You can NOT use the linear algebra lib 'numpy.linalg' of numpy.         #
        # Do not forget the self.reg_ item.                                       #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        assert self.coef_ is not None
        assert self.intercept_ is not None
        return self

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array, shape (n_samples, n_targets)
            Returns predicted values.
        """
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted values in y_pred.            #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return y_pred