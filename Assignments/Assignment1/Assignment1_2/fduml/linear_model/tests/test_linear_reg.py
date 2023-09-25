"""
test linear regression
"""

import numpy as np
from numpy.testing import assert_array_almost_equal
from fduml.linear_model import LinearRegression

def test_linear_regression():
    # Test LinearRegression on a simple dataset.
    # a simple dataset
    X = np.array([[1], [2]])
    Y = np.array([[1], [2]])

    reg = LinearRegression()
    reg.fit(X, Y)

    assert_array_almost_equal(reg.coef_, [1])
    assert_array_almost_equal(reg.intercept_, [0])
    assert_array_almost_equal(reg.predict(X), Y)

    # test it also for degenerate input
    X = np.array([[1]])
    Y = np.array([[0]])

    reg = LinearRegression()
    reg.fit(X, Y)
    assert_array_almost_equal(reg.coef_, [0])
    assert_array_almost_equal(reg.intercept_, [0])
    assert_array_almost_equal(reg.predict(X), Y)

