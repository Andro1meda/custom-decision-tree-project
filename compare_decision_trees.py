import time
import tracemalloc

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

# Try importing memory_profiler to measure memory usage
try:
    from memory_profiler import memory_usage

    USE_MEMORY_PROFILER = True
except ImportError:
    USE_MEMORY_PROFILER = False

from custom_decision_tree import DecisionTreeClassifier, predict


def train_custom_decision_tree(X_train, y_train, sample_min_split, max_depth):
    tracemalloc.start()

    #  Train the custom decision tree; measure memory usage if memory_profiler is available.

    def train_func():
        clf = DecisionTreeClassifier(
            sample_min_split=sample_min_split, max_depth=max_depth)
        return clf.grow_tree(X_train, y_train)

    if USE_MEMORY_PROFILER:
        mem_usage_list = memory_usage(
            (train_func,), max_iterations=1, interval=0.1)
        tree_root = train_func()  # Actually build the tree
        return tree_root, max(mem_usage_list)
    else:
        start_snapshot = tracemalloc.take_snapshot()
        clf = DecisionTreeClassifier(
            sample_min_split=sample_min_split, max_depth=max_depth)
        tree_root = clf.grow_tree(X_train, y_train)
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        peak_memory = sum(stat.size for stat in stats) / (1024 * 1024)  # Convert storage to MB
        return tree_root, peak_memory


def evaluate_model(X_train, y_train, X_test, y_test, max_depth, sample_min_split, test_size):
    # Train and evaluate both custom and sklearn decision trees, returning performance metrics.
    # Custom Decision Tree
    start_time = time.time()
    custom_tree_root, custom_memory = train_custom_decision_tree(
        X_train, y_train, sample_min_split, max_depth
    )
    custom_train_time = time.time() - start_time

    start_time = time.time()
    custom_preds = predict(X_test, custom_tree_root)
    custom_prediction_time = time.time() - start_time

    custom_acc = accuracy_score(y_test, custom_preds)
    custom_prec = precision_score(
        y_test, custom_preds, average='macro', zero_division=0)
    custom_rec = recall_score(y_test, custom_preds,
                              average='macro', zero_division=0)
    custom_f1 = f1_score(y_test, custom_preds,
                         average='macro', zero_division=0)

    # Sklearn Decision Tree
    sklearn_tree = SklearnDecisionTree(
        criterion='gini',
        max_depth=max_depth,
        min_samples_split=sample_min_split,
        random_state=42
    )

    start_time = time.time()
    if USE_MEMORY_PROFILER:
        def sklearn_train_func():
            sklearn_tree.fit(X_train, y_train)

        sklearn_mem_usage_list = memory_usage(
            (sklearn_train_func,), max_iterations=1, interval=0.1)
        sklearn_train_func()
        sklearn_train_time = time.time() - start_time
        sklearn_memory = max(sklearn_mem_usage_list)
    else:
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        sklearn_tree.fit(X_train, y_train)
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        sklearn_train_time = time.time() - start_time
        sklearn_memory = sum(stat.size for stat in stats) / (1024 * 1024)

    start_time = time.time()
    sklearn_preds = sklearn_tree.predict(X_test)
    sklearn_prediction_time = time.time() - start_time

    sklearn_acc = accuracy_score(y_test, sklearn_preds)
    sklearn_prec = precision_score(
        y_test, sklearn_preds, average='macro', zero_division=0)
    sklearn_rec = recall_score(
        y_test, sklearn_preds, average='macro', zero_division=0)
    sklearn_f1 = f1_score(y_test, sklearn_preds,
                          average='macro', zero_division=0)
    # get table of results
    results = {
        'max_depth': max_depth,
        'sample_min_split': sample_min_split,
        'test_size': test_size,

        'custom_accuracy': custom_acc,
        'custom_precision': custom_prec,
        'custom_recall': custom_rec,
        'custom_f1': custom_f1,
        'custom_train_time': custom_train_time,
        'custom_prediction_time': custom_prediction_time,
        'custom_memory_usage': custom_memory,

        'sklearn_accuracy': sklearn_acc,
        'sklearn_precision': sklearn_prec,
        'sklearn_recall': sklearn_rec,
        'sklearn_f1': sklearn_f1,
        'sklearn_train_time': sklearn_train_time,
        'sklearn_prediction_time': sklearn_prediction_time,
        'sklearn_memory_usage': sklearn_memory
    }
    return results


def main():
    # Importing datasets compares custom vs sklearn DecisionTree.
    from ucimlrepo import fetch_ucirepo

    datasets = [53, 109, 17]

    for dataset in datasets:
        print(f"Loading dataset {dataset}")
        _data = fetch_ucirepo(id=dataset)

        X = _data.data.features
        y = _data.data.targets

        # Convert X, y to numpy arrays
        X = X.to_numpy()
        y = np.array(y)

        test_sizes = [0.1, 0.3, 0.5, 0.7]

        all_results = []

        for test_size in test_sizes:
            # Then split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            max_depths = [1, 2, 3, 5, 10]
            sample_splits = [2, 3, 4, 5]

            for md in max_depths:
                for sms in sample_splits:
                    res = evaluate_model(X_train, y_train, X_test, y_test, md, sms, test_size)
                    all_results.append(res)

        # Save to CSV
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(f"results_custom_vs_sklearn_dataset_{dataset}.csv", index=False)
        print("Results saved to 'results_custom_vs_sklearn.csv'")


main()
