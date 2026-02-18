import pytest
import numpy as np

from custom_decision_tree import DecisionTreeClassifier, predict
from ucimlrepo import fetch_ucirepo


@pytest.fixture
def wine_dataset():
    # importing wine dataset for test as a small dataset
    wine = fetch_ucirepo(id=109)
    X = wine.data.features
    y = wine.data.targets

    # turning dataset into numpy
    X = X.to_numpy()
    y = y.to_numpy().ravel()  # flattening into a 1D array

    return X, y


# checking the structure of the wine dataset

def test_fetch_ucirepo_structure():
    wine = fetch_ucirepo(id=109)
    print(wine.data.features)  # Should show features
    print(wine.data.targets)  # Should show labels


# tests

def test_tree_creation_with_wine(wine_dataset):  # testing tree creation
    X, y = wine_dataset
    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=3)
    root = clf.grow_tree(X, y)
    assert root is not None
    # check whether root node exists then assert root node as not leaf node
    assert root.is_leaf_node() is False, "The root is not a leaf node"


def test_tree_prediction(wine_dataset):
    X, y = wine_dataset
    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=3)
    root = clf.grow_tree(X, y)
    preds = predict(X, root)  # list of predicted labels
    correct = np.sum(y == preds)  # counting the no. of correct predictions
    assert correct >= len(y) * 0.9  # setting threshold of 90% prediction accuracy


def test_one_class(wine_dataset):  # testing whether it predicts correctly when training with only one class
    X, y = wine_dataset
    X = X[y == 1]  # filtering only with "1" class
    y = y[y == 1]

    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=3)
    root = clf.grow_tree(X, y)
    assert root.is_leaf_node(), "Tree with one class is a leaf node"
    assert root.value == 1, "Tree with one class"


def test_small_subset(wine_dataset):  # testing with a smaller subset of the iris dataset
    X, y = wine_dataset
    X = X[:5]  # choosing first 5 samples
    y = y[:5]
    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=3)
    root = clf.grow_tree(X, y)
    preds = predict(X, root)  # list of predicted labels
    assert np.all(preds == y), "Predicting with a smaller subset"


def test_high_max_depth(wine_dataset):
    X, y = wine_dataset
    clf = DecisionTreeClassifier(sample_min_split=2, max_depth=25)
    root = clf.grow_tree(X, y)
    preds = predict(X, root)
    assert np.all(preds == y), "Predicting with higher max depth"

