import pytest
import numpy as np

from custom_decision_tree import DecisionTreeClassifier, predict
from ucimlrepo import fetch_ucirepo


@pytest.fixture
def breast_cancer_dataset():
    # importing wine dataset for test as a small dataset
    breast_cancer = fetch_ucirepo(id=17)
    X = breast_cancer.data.features
    y = breast_cancer.data.targets

    # turning dataset into numpy
    X = X.to_numpy()
    y = y.to_numpy().ravel()  # flattening into a 1D array

    return X, y


# checking the structure of the wine dataset

def test_fetch_ucirepo_structure():
    breast_cancer = fetch_ucirepo(id=17)
    print(breast_cancer.data.features)  # Should show features
    print(breast_cancer.data.targets)  # Should show labels


# tests

def test_tree_creation_with_breast_cancer(breast_cancer_dataset):  # testing tree creation
    X, y = breast_cancer_dataset
    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=3)
    root = clf.grow_tree(X, y)
    assert root is not None
    # check whether root node exists then assert root node as not leaf node
    assert root.is_leaf_node() is False, "The root is not a leaf node"


def test_tree_prediction(breast_cancer_dataset):
    X, y = breast_cancer_dataset
    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=3)
    root = clf.grow_tree(X, y)
    preds = predict(X, root)  # list of predicted labels
    correct = np.sum(y == preds)  # counting the no. of correct predictions
    assert correct >= len(y) * 0.9  # setting threshold of 90% prediction accuracy


def test_one_class(breast_cancer_dataset):  # testing whether it predicts correctly when training with only one class
    X, y = breast_cancer_dataset
    X = X[y == "M"]  # filtering only with "M" class
    y = y[y == "M"]

    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=3)
    root = clf.grow_tree(X, y)
    assert root.is_leaf_node(), "Tree with one class is a leaf node"
    assert root.value == "M", "Tree with one class"


def test_small_subset(breast_cancer_dataset):  # testing with a smaller subset of the iris dataset
    X, y = breast_cancer_dataset
    X = X[:6]  # choosing first 6 samples
    y = y[:6]
    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=3)
    root = clf.grow_tree(X, y)
    preds = predict(X, root)  # list of predicted labels
    assert np.all(preds == y), "Predicting with a smaller subset"


def test_high_max_depth(breast_cancer_dataset):
    X, y = breast_cancer_dataset
    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=25)
    root = clf.grow_tree(X, y)
    preds = predict(X, root)
    assert np.all(preds == y), "Predicting with higher max depth"

