import numpy as np
from collections import Counter

from .criterion import get_criterion_function

class node:
    def __init__(self,
                 feat_idx=None,
                 threshold=None,
                 split_score=None,
                 left=None,
                 right=None,
                 value=None,
                 leaf_num=None):
        """
        :param feat_idx: attribute idx when splitting
        :param threshold: threshold of the attribute when splitting
        :param split_score: score like info gain when splitting
        :param left: left tree
        :param right: right tree
        :param value: class id
        :param leaf_num: the leaf node number under this node
        """
        self.feat_idx = feat_idx
        self.threshold = threshold
        self.split_score = split_score
        self.value = value
        self.left = left
        self.right = right
        self.leaf_num = leaf_num


class DecisionTreeClassifier(object):
    """A decision tree classifier.

    Parameters
    ----------
    criterion : {"info_gain", "info_gain_ratio", "gini", "error_rate"}, default="info_gain"
        The function to measure the quality of a split.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    max_features : int, default=None
        The number of features to consider when looking for the best split.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    min_impurity_split: float, default=0.0
        The minimum value of impurity required to split an internal node.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
    """
    def __init__(self,
                 criterion="info_gain",
                 splitter="best",
                 max_depth=None,
                 max_features=None,
                 min_samples_split=2,
                 min_impurity_split=0.0,
                 random_state=None):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = float("inf") if max_depth is None else max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_impurity_split = min_impurity_split
        self.random_state = random_state

        self.root = None
        self.feature_importances_ = None
        self.feature_scores_ = None

        self.tree_leaf_num = 0
        self.tree_depth = 0

        self._score_func = get_criterion_function(criterion)

    def fit(self, X, y):
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples, )
            The target values (class labels) as integers.
        """
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.root = self._build_tree(X, y)

        # normalize feature scores
        if self.feature_scores_.sum() != 0.0:
            self.feature_importances_ = (
                    self.feature_scores_ / self.feature_scores_.sum())
        else:
            self.feature_importances_ = np.zeros(X.shape[1], dtype=float)

    def _build_tree(self, X, y, curr_depth=1):
        """
        Build the tree recursively

        curr_depth: current depth
        """
        n_samples, n_feats = X.shape
        self.feature_scores_ = np.zeros(n_feats, dtype=float)

        split = None
        split_score = 0
        if n_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            split, split_score = self._split(X, y, self.splitter)

        leaf_num_before = self.tree_leaf_num
        if split_score > self.min_impurity_split:
            left = self._build_tree(split["l_X"], split["l_y"], curr_depth + 1)
            right = self._build_tree(split["r_X"], split["r_y"], curr_depth + 1)
            self.feature_scores_[split["feat_idx"]] += split_score
            return node(feat_idx=split["feat_idx"], threshold=split["threshold"],
                        split_score=split_score, left=left, right=right,
                        leaf_num=self.tree_leaf_num - leaf_num_before)
        else:
            leaf_val = self._aggregation_func(y)
            self.tree_leaf_num += 1
            if curr_depth > self.tree_depth:
                self.tree_depth = curr_depth
            return node(split_score=split_score, value=leaf_val, leaf_num=1)

    def _split(self, X, y, splitter="best"):
        """
        Split the node

        Returns
        -------
        best_split: a dict {"feat_idx": col, "threshold": thr,
                        "l_X": l_Xy[:, :n_feats], "r_X": r_Xy[:, :n_feats],
                        "l_y": l_y, "r_y": r_y}

        max_score: max value of the criterion
        """
        Xy = np.concatenate((X, y), axis=1)
        n_feats = X.shape[1]

        max_score = 0.0
        best_split = None

        k = self._get_n_feats(self.max_features, n_feats)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        cols = np.random.choice(range(n_feats), k, replace=False)

        n_sample = X.shape[0]

        for c in cols:
            if splitter == "random":
                thr_i = np.random.randint(0, n_sample)
                thr = X[thr_i, c]
                l, r = self._thr_split(X, c, thr, n_sample)
                l_y = y[l]
                r_y = y[r]

                score = self._score_func(y, l_y, r_y)
                if score > max_score:
                    l_Xy = X[l, :]
                    r_Xy = X[r, :]
                    best_split = {"feat_idx": c, "threshold": thr,
                                  "l_X": l_Xy, "r_X": r_Xy,
                                  "l_y": l_y, "r_y": r_y}
                    max_score = score
            else:
                for i in range(n_sample):
                    thr = X[i, c]

                    l, r = self._thr_split(X, c, thr, n_sample)
                    l_y = y[l]
                    r_y = y[r]

                    score = self._score_func(y, l_y, r_y)
                    if score > max_score:
                        l_Xy = X[l, :]
                        r_Xy = X[r, :]
                        best_split = {"feat_idx": c, "threshold": thr,
                                      "l_X": l_Xy, "r_X": r_Xy,
                                      "l_y": l_y, "r_y": r_y}
                        max_score = score
        return best_split, max_score

    @staticmethod
    def _thr_split(X, c, thr, n_sample):
        """ Split data by threshold """

        left_data = []
        right_data = []

        for i in range(n_sample):
            if X[i, c] < thr:
                left_data.append(i)
            else:
                right_data.append(i)

        return left_data, right_data

    @staticmethod
    def _get_n_feats(max_feats, n_feats):
        """ Get k features from n_features """

        if isinstance(max_feats, int):
            return max_feats
        elif isinstance(max_feats, float):
            return int(max_feats * n_feats)
        elif isinstance(max_feats, str):
            if max_feats == "sqrt":
                return int(np.sqrt(n_feats))
            elif max_feats == "log2":
                return int(np.log2(n_feats + 1))
        return n_feats

    def predict(self, X):
        """ Predict the label of X """

        if X.ndim == 1:
            return self._predict_sample(X)
        else:
            return np.array([self._predict_sample(sample) for sample in X])

    def _aggregation_func(self, y):
        """ Get the label of a leaf node """
        res = Counter(y.reshape(-1))
        return int(res.most_common()[0][0])

    def _predict_sample(self, x, node=None):
        """ Predict which node the sample is in """
        if node is None:
            node = self.root

        if node.value is not None:
            return node.value

        feat = x[node.feat_idx]
        node = node.left if feat < node.threshold else node.right
        return self._predict_sample(x, node=node)

    @staticmethod
    def _divide(data, col, thr):
        """ Divide the data by the threshold of a column """
        mask = data[:, col] < thr
        return data[mask], data[~mask]
