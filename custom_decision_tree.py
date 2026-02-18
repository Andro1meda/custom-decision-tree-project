import numpy as np


class Node:
    def __init__(self, attributes=None, left=None, right=None, threshold=None, value=None):
        self.attributes = attributes  # Which column/feature used for splitting
        self.left = left  # Left
        self.right = right  # Right
        self.threshold = threshold  # Threshold value for splitting
        self.value = value  # Class label if leaf

    def is_leaf_node(self):
        return self.left is None and self.right is None


class DecisionTreeClassifier:
    def __init__(self, sample_min_split=1, max_depth=3):
        # Constructing custom Decision Tree
        # sample_min_split: minimum number of samples required to split
        # max_depth: maximum depth of the tree
        self.sample_min_split = sample_min_split
        self.max_depth = max_depth
        self.root = None

    def gini_impurity(self, y):
        # Calculate Gini impurity for labels y.
        _, label_counts = np.unique(y, return_counts=True)
        N = len(y)
        gini = 1.0 - sum((count / N) ** 2 for count in label_counts)
        return gini

    def _best_split(self, X, y):
        # Find the best attribute and threshold to split on.
        # Return:
        # attribute_index, threshold, mask_left, mask_right
        n_samples = len(X)
        n_attributes = len(X[0])

        if n_samples <= 1:
            return None, None, None, None

        # Current impurity
        gini_val = self.gini_impurity(y)
        best_gini_gain = 0.0
        best_attribute_index = None
        best_threshold = None
        best_left = None
        best_right = None

        for attribute_index in range(n_attributes):
            # Unique values for this attribute
            unique_vals = np.unique(X[:, attribute_index])
            if len(unique_vals) == 1:
                continue

            for threshold in unique_vals:
                mask_left = (X[:, attribute_index] <= threshold)
                mask_right = (X[:, attribute_index] > threshold)

                if np.sum(mask_left) == 0 or np.sum(mask_right) == 0:
                    continue

                # Calculate impurity
                left_impurity = self.gini_impurity(y[mask_left])
                right_impurity = self.gini_impurity(y[mask_right])

                weight_left = np.sum(mask_left) / n_samples
                weight_right = np.sum(mask_right) / n_samples
                weighted_impurity = weight_left * left_impurity + weight_right * right_impurity

                # Gini gain
                gini_gain = gini_val - weighted_impurity

                # Update best split
                if gini_gain > best_gini_gain:
                    best_gini_gain = gini_gain
                    best_attribute_index = attribute_index
                    best_threshold = threshold
                    best_left = mask_left
                    best_right = mask_right

        if best_attribute_index is None:
            return None, None, None, None

        return best_attribute_index, best_threshold, best_left, best_right

    def grow_tree(self, X, y, depth=0):

        # Recursively build the decision tree.

        unique_labels = np.unique(y)

        # If only one label, create a leaf
        if len(unique_labels) == 1:
            return Node(value=unique_labels[0])

        # If max depth reached or not enough samples then create a leaf with majority label
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.sample_min_split:
            majority_label = unique_labels[np.argmax(
                [np.sum(y == label) for label in unique_labels])]
            return Node(value=majority_label)

        attribute_index, threshold, mask_left, mask_right = self._best_split(
            X, y)

        # Becomes leaf node if no further splitting possible
        if attribute_index is None:
            majority_label = unique_labels[np.argmax(
                [np.sum(y == label) for label in unique_labels])]
            return Node(value=majority_label)

        # Recursively build subtrees
        left_node = self.grow_tree(X[mask_left], y[mask_left], depth + 1)
        right_node = self.grow_tree(X[mask_right], y[mask_right], depth + 1)
        return Node(attributes=attribute_index,
                    threshold=threshold,
                    left=left_node,
                    right=right_node)


def predict_one(sample, node):
    # Predict the label for a single sample by traversing the tree
    if node.is_leaf_node():
        return node.value
    if sample[node.attributes] <= node.threshold:
        return predict_one(sample, node.left)
    else:
        return predict_one(sample, node.right)


def predict(X, node):
    # Predict for multiple samples
    return np.array([predict_one(sample, node) for sample in X])

