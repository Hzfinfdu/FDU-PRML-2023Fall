"""
test knn
"""

import numpy as np
from numpy.testing import assert_array_equal
from fduml.neighbors import KNeighborsClassifier

def test_kneighbors_classifier(
        num_train=40,
        n_features=5,
        num_test=10,
        n_neighbors=5,
        random_state=0,
):
    # Test k-neighbors classification
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(num_train, n_features) - 1
    y = ((X**2).sum(axis=1) < 0.5).astype(int)
    epsilon = 1e-5 * (2 * rng.rand(1, n_features) - 1)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors,num_loops=2)
    knn.fit(X, y)
    y_pred = knn.predict(X[:num_test] + epsilon)
    assert_array_equal(y_pred, y[:num_test])

    knn = KNeighborsClassifier(n_neighbors=n_neighbors,num_loops=1)
    knn.fit(X, y)
    y_pred = knn.predict(X[:num_test] + epsilon)
    assert_array_equal(y_pred, y[:num_test])

    knn = KNeighborsClassifier(n_neighbors=n_neighbors,num_loops=0)
    knn.fit(X, y)
    y_pred = knn.predict(X[:num_test] + epsilon)
    assert_array_equal(y_pred, y[:num_test])