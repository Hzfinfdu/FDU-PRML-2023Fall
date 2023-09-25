"""
K-nearest Neighbor Algorithm
"""
import numpy as np

class KNeighborsClassifier(object):
    """
    Classifier implementing the k-nearest neighbors vote with L2 distance.

    Parameters
    ----------
    n_neighbors : int, default=5
        The number of nearest neighbors that vote for the predicted labels.

    num_loops: int, default=0
        Determines which implementation to use to compute distances between training points and testing points.
    """

    def __init__(self, n_neighbors=5, num_loops = 0):
        self.k = n_neighbors
        self.num_loops = num_loops

        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Fit the k-nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        X : {array-like matrix} of shape (num_train, n_features)
            Training data.
        y : {array-like matrix} of shape (num_train, n_outputs)
            Target values.

        Returns
        -------
        self : KNeighborsClassifier
            The fitted k-nearest neighbors classifier.
        """
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. The simplest way is to store X and y with        #
        # self.X_train and self.y_train directly                                  #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : {array-like matrix} of shape (n_test, n_features)
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_test, n_outputs)
            Class labels for each data sample.
        """

        if self.num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif self.num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif self.num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % self.num_loops)

        return self.predict_labels(dists)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Parameters
        ----------
        - X: A numpy array of shape (num_test, n_features) containing test data.

        Returns
        -------
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

                pass

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.
        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.
        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Parameters
        ----------
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns
        -------
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length self.k storing the labels of the self.k nearest
            # neighbors to the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the self.k nearest neighbors of       #
            # the ith testing point, and use self.y_train to find the               #
            # labels of these neighbors. Store these labels in closest_y.           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            ##########################################################################
            # TODO:                                                                  #
            # Now that you have found the labels of the self.k nearest neighbors,    #
            # you need to find the most common label in the list closest_y of        #
            # labels. Store this label in y_pred[i]. Break ties by choosing          #
            # the smaller label.                                                     #
            ##########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred