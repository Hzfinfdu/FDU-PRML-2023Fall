"""
decision tree test
"""

import numpy as np
from numpy.testing import assert_array_equal
from fduml.tree import DecisionTreeClassifier

def test_dt_classification():
	X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
	y = np.array([-1, -1, -1, 1, 1, 1])

	T = np.array([[-1, -1], [2, 2], [3, 2]])

	criterion = "info_gain"
	true_result = np.array([-1, 1, 1])
	dt_clf = DecisionTreeClassifier(criterion=criterion, random_state=0)
	dt_clf.fit(X, y)
	print(X)
	assert_array_equal(dt_clf.predict(T), true_result, "Failed with {}".format(criterion))

	criterion = "info_gain_ratio"
	true_result = np.array([-1, 1, 1])
	dt_clf = DecisionTreeClassifier(criterion=criterion, random_state=0)
	dt_clf.fit(X, y)
	assert_array_equal(dt_clf.predict(T), true_result, "Failed with {}".format(criterion))

	criterion = "gini"
	true_result = np.array([-1, 1, 1])
	dt_clf = DecisionTreeClassifier(criterion=criterion, random_state=0)
	dt_clf.fit(X, y)
	assert_array_equal(dt_clf.predict(T), true_result, "Failed with {}".format(criterion))

	criterion = "error_rate"
	true_result = np.array([-1, 1, 1])
	dt_clf = DecisionTreeClassifier(criterion=criterion, random_state=0)
	dt_clf.fit(X, y)
	assert_array_equal(dt_clf.predict(T), true_result, "Failed with {}".format(criterion))


if __name__ == '__main__':
	test_dt_classification()
