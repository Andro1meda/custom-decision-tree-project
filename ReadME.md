# Decision Tree Classifier: Custom vs Scikit-learn Comparison

## Project Overview

This project compares a custom-built Decision Tree Classifier with the Scikit-learn DecisionTreeClassifier to evaluate:

**Machine Learning performance**
- Accuracy
- Precision
- Recall
- F1-score

**Computational efficiency**
- Training time
- Prediction time
- Memory usage

The goal was to determine which implementation performs more efficiently across multiple datasets of varying sizes.

## Datasets Used

The following datasets from the UCI Machine Learning Repository were used:
- Iris (small dataset, 4 features, 3 classes)
- Wine (medium dataset, 13 features, 3 classes)
- Breast Cancer Wisconsin (Diagnostic) (large dataset, 30 features, 2 classes)

## Methodology

**Custom Decision Tree**
- Implemented from scratch in Python
- Used Gini Impurity for splitting

_**Hyperparameters:**_
- max_depth
- min_samples_split

**Scikit-learn Implementation**
- DecisionTreeClassifier
- Same hyperparameters applied for fair comparison

_Hyperparameters were systematically varied:_
- _max_depth: 1–10_
- _min_samples_split: 2–5_
- _Test sizes: 10%, 30%, 50%, 70%_

**Statistical tests:**
- Paired t-tests
- Two-way ANOVAs

## Key Findings

**Machine Learning Performance**
- Scikit-learn performed better overall across Iris and Wine datasets.
- Custom tree performed slightly better on the Breast Cancer dataset.
- max_depth had a statistically significant effect on performance.

**Computational Performance**
- Scikit-learn trained significantly faster and used less memory
- Similar prediction time to custom implementation

## Tools Used

**Python:**
- NumPy
- Pandas
- Scikit-learn
- memory_profiler
- tracemalloc

**R:**
- tidyverse
- broom
- ggplot2
- patchwork

## Conclusion

The Scikit-learn implementation demonstrated superior computational efficiency and generally stronger predictive performance.

However, the custom implementation achieved comparable results and in some cases outperformed Scikit-learn at specific tree depths.

*_More information and figures are found in the full report "Project.pdf", "Project - Copy.Rmd" includes code snippets for the figures and statistical tests_*
