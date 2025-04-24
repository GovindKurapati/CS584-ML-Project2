import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class TreeNode:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        *,
        value=None,
        leaf_id=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # for leaf: the prediction (residual fit or gamma)
        self.leaf_id = leaf_id  # unique ID for each leaf region


class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self._leaf_counter = 0

    def fit(self, X, y):
        self._leaf_counter = 0
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        # Stop conditions
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or n_samples <= self.min_samples_leaf
        ):
            leaf_value = np.mean(y)
            node = TreeNode(value=leaf_value, leaf_id=self._leaf_counter)
            self._leaf_counter += 1
            return node

        # Find best split
        best_feat, best_thresh, best_mse = None, None, np.inf
        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                right_mask = ~left_mask
                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue
                y_left, y_right = y[left_mask], y[right_mask]
                mse = ((y_left - y_left.mean()) ** 2).sum() + (
                    (y_right - y_right.mean()) ** 2
                ).sum()
                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat
                    best_thresh = thr

        # If no valid split found, make leaf
        if best_feat is None:
            leaf_value = np.mean(y)
            node = TreeNode(value=leaf_value, leaf_id=self._leaf_counter)
            self._leaf_counter += 1
            return node

        # Split and recurse
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return TreeNode(
            feature_index=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
        )

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def apply(self, X):
        """Return leaf_id for each sample."""
        return np.array([self._leaf_id_one(x, self.root) for x in X])

    def _leaf_id_one(self, x, node):
        if node.value is not None:
            return node.leaf_id
        if x[node.feature_index] <= node.threshold:
            return self._leaf_id_one(x, node.left)
        else:
            return self._leaf_id_one(x, node.right)


class GradientBoostingClassifier:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.gammas = []
        self.F0 = None

    def fit(self, X, y):
        y = np.array(y, dtype=float)
        n_samples = y.shape[0]
        # Initial log-odds
        pos = np.clip(y.sum(), 1e-6, n_samples - 1e-6)
        self.F0 = np.log(pos / (n_samples - pos))

        # Initialize model predictions
        F = np.full(n_samples, self.F0)

        for m in range(self.n_estimators):
            # Probabilities and residuals
            p = sigmoid(F)
            residuals = y - p

            # Fit tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X, residuals)

            # Find region assignments
            leaf_ids = tree.apply(X)
            gamma_map = {}

            # Compute gamma for each leaf region
            for leaf in np.unique(leaf_ids):
                mask = leaf_ids == leaf
                numer = residuals[mask].sum()
                denom = (p[mask] * (1 - p[mask])).sum()
                gamma_map[leaf] = numer / (denom + 1e-12)

            # Update F
            update = np.array([gamma_map[leaf] for leaf in leaf_ids])
            F += self.learning_rate * update

            # Store tree and its gamma_map
            self.trees.append(tree)
            self.gammas.append(gamma_map)

    def predict_probability(self, X):
        # Start from initial
        F = np.full(X.shape[0], self.F0)
        for tree, gamma_map in zip(self.trees, self.gammas):
            leaf_ids = tree.apply(X)
            update = np.array([gamma_map[leaf] for leaf in leaf_ids])
            F += self.learning_rate * update
        return sigmoid(F)

    def predict(self, X):
        proba = self.predict_probability(X)
        return (proba >= 0.5).astype(int)

    def evaluate_accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
