import pytest
import numpy as np

from custom_decision_tree import DecisionTreeClassifier, predict
from ucimlrepo import fetch_ucirepo


@pytest.fixture
def iris_dataset():
    # importing iris dataset for test as a small dataset
    iris = fetch_ucirepo(id=53)
    X = iris.data.features
    y = iris.data.targets

    # turning dataset into numpy
    X = X.to_numpy()
    y = y.to_numpy().ravel()  # flattening into a 1D array

    return X, y


# checking the structure of the iris dataset

def test_fetch_ucirepo_structure():
    iris = fetch_ucirepo(id=53)
    print(iris.data.features)  # Should show features
    print(iris.data.targets)  # Should show labels


# tests

def test_tree_creation_with_iris(iris_dataset):  # testing tree creation
    X, y = iris_dataset
    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=3)
    root = clf.grow_tree(X, y)
    assert root is not None
    # check whether root node exists then assert root node as not leaf node
    assert root.is_leaf_node() is False, "The root is not a leaf node"


def test_tree_prediction(iris_dataset):
    X, y = iris_dataset
    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=3)
    root = clf.grow_tree(X, y)
    preds = predict(X, root)  # list of predicted labels
    correct = np.sum(y == preds)  # counting the no. of correct predictions
    assert correct >= len(y) * 0.9  # setting threshold of 90% prediction accuracy


def test_one_class(iris_dataset):  # testing whether it predicts correctly when training with only one class
    X, y = iris_dataset
    X = X[y == "Iris-setosa"]  # filtering only with "iris-setosa" class
    y = y[y == "Iris-setosa"]

    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=3)
    root = clf.grow_tree(X, y)
    assert root.is_leaf_node(), "Tree with one class is a leaf node"
    assert root.value == "Iris-setosa", "Tree with one class"


def test_small_subset(iris_dataset):  # testing with a smaller subset of the iris dataset
    X, y = iris_dataset
    X = X[:4]  # choosing first 4 samples
    y = y[:4]
    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=3)
    root = clf.grow_tree(X, y)
    preds = predict(X, root)  # list of predicted labels
    assert np.all(preds == y), "Predicting with a smaller subset"


def test_high_max_depth(iris_dataset):
    X, y = iris_dataset
    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=25)
    root = clf.grow_tree(X, y)
    preds = predict(X, root)
    assert np.all(preds == y), "Predicting with higher max depth"


