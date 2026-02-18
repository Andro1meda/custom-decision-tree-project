
Title: SCC.461 Final Assignment
Date: 2025-01-09


```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE, message = FALSE)
library(tidyverse)
library(broom)
library(ggplot2)
library(patchwork)
```

# 1. Abstract

This report compared a custom implementation of a Decision Tree Classifier against a SciKit-learn decision tree, and evaluated their performance against computational (training time, prediction time and memory usage) and machine learning aspects (accuracy, precision, recall and f1 score). The aim of this investigation was to determine which decision tree had the best performance metric results and thus was the most efficient. Both implementations were used using three well known datasets: "Iris", "Wine" and "Breast Cancer Wisconsin (Diagnostic)". Overall, the Sklearn decision tree performed better showing higher metric scores on computational and machine learning aspects compared to the custom decision tree. However, when comparing accuracy against the hyper-parameter of maximum depth of the tree, the custom decision tree showed greater accuracy scores than the Sklearn decision tree. 


# 2. Introduction

## 2.1 Decision Trees

Decision trees are one of the most widely used and most implemented non-parametric supervised algorithms in machine learning and are used in many other aspects such as for decision analysis and for artificial intelligence. Decision trees, although can be used for both classification and regression tasks, this project will only focus on decision tree classifiers. 

Decision trees aim to predict a target variable from input variables and attributes. To understand how this works, we first need to know how it is structured. A decision tree has a hierarchical structure with "nodes". Nodes represent the decisions based on attributes which will split off into branches leading to "decision nodes" or "internal nodes", this process can be repeated multiple times, where the "internal nodes" or "decision nodes" will become the new "root node" thus forming a sub-tree. This process will continue until either meeting stopping criteria or until the subset data is pure. The last nodes are "leaf nodes" representing the actual output. 


Gini Impurity was used as a metric to determine the best splitting point at each node; it measures how pure a dataset is using this formula.

$$
\text{Gini Index} = 1 - \sum_{i=1}^{n} p_i^2

$$

Where $p_i$ is the proportion of data points in class $i$ and $n$ is the total number of classes,  

Although Entropy can be another metric to determine the best split that works quite similar to Gini Index, Gini index was more ideal as it is less intensive and is much faster to calculate. However, due to entropy having more complex calculation through logarithmic functions, it is more robust and the results obtained from it are better than Gini(GeeksforGeeks, 2024; Kaushik, 2023).

## 2.2 Scikit-learn

Scikit-learn is a widely used open-sourced machine learning library in Python, providing efficient tools for not only regression, forests but classification, which helps to provide machine learning implementations such as Decision Tree Classifier used in this project to compare against a custom implementation. 
One of the features of the Scikit-learn Decision Tree Classifier is that it defaults to using Gini Index to determine the best split, however there is the option to change it do Entropy (1.10. Decision Trees, n.d.; What Is Sklearn? | Domino Data Lab, n.d.).


## 2.3 Datasets

The data sets chosen were Iris, Wine and Breast Cancer Wisconsin (Diagnostic) picked out from the IC Irvin Machine Learning Repository (UCI Machine Learning Repository, n.d.). These datasets were mainly picked out due to the size of the database, with varying number of features. 

The Iris dataset (Fisher, 1936) is much smaller than the others with four attributes depicting the features of the plant (sepal length, sepal width, petal width and petal length). From those four attributes it will determine whether it would be classified into one of three classes of Iris plant: Iris Setosa, Iris Versicolour and Iris Virginica. 

The Wine dataset has 13 attributes and 3 classes of different wine cultivars, a bigger data set than Iris but not having the problem of having too many attributes thus is a good data set when first testing new decision tree classifiers due to not having challenging class structure. Finally, to challenge the classifier with a bigger dataset, Breast Cancer Wisconsin (Diagnostic) was used, with 30 attributes and 2 classes of diagnosis (malignant or benign).

## 2.4 Decision Tree Performance Measures


The aim of this project was to evaluate the machine learning and computational aspects of a custom and sklearn decision tree implementation. 

Machine learning metrics such as accuracy, precision, recall and f1-scores were measured, where f1 score is the balance of recall and precision scores. Higher scores indicate better decision tree performance. 

Computational aspects such as memory usage, training and prediction time were also measured to explore the efficiency of the models and compare them against each other. Lower scores on computational measures indicate better decision tree performance in these aspects.



# 3. Methods

This section provides an overview of the methods that went into the implementation process for both the custom decision tree and sklearn DecisionTreeClassifier. Focusing on how the tree was configurated, the training and prediction of the tree.

## 3.1 Tree Configurations

When configurating the decision tree, hyper-parameters were set to control for over-fitting and under-fitting the data and making it as accurate as possible.

The hyper-parameter "max_depth" refers to the maximum depth of the tree, it was used to control for the overall complexity of the tree. However this would allow for over-fitting if not defined as the nodes will keep splitting until all leaf nodes are pure (Importance of Decision Tree Hyperparameters on Generalization — Scikit-learn Course, n.d.). This project varied the tree depth from shallow (1) to very deep (10) to explore explore how this will affect the performance metrics (specifically accuracy) as the deeper the tree grows the more complex it becomes (Decision Tree Sklearn -Depth of Tree and Accuracy, n.d.).

Another hyper-parameter, "sample_min_split", which refers to the minimum samples required to split decision nodes, was used. This means that if the data in the decision node is less than the minimum samples required, the node will become a leaf node. This hyper-parameter ensures the samples decide which split is considered, where large numbers prevent the tree from learning the data and small numbers mean the tree is an risk for overfitting (Importance of Decision Tree Hyperparameters on Generalization — Scikit-learn Course, n.d.). Thus, this project also varied the minimum sample size from a range of 2 to 5 to evaluate its effect on the accuracy of classifying.

These hyper-parameters are in place to ensure the tree performs well, as a shallow tree depth and fewer leaf nodes have been shown to be indicators of an efficient decision tree (Aaboub et al., 2023). 

Test size was also varied when evaluating the decision trees. Test size refers to the portion of the dataset that is being tested, this ranged from 10%, 30%, 50% and 70%, to explore if the proportion of the dataset being tested affects the machine learning metrics, specifically accuracy results, of both decision tree implementations. For loops were used to test the combination of each of the hyper-parameters. 


## 3.2 Fitting and Testing the Tree

### Custom Decision Tree

Initially, when training the custom decision tree, the best split had to be determined. This was done through recursively splitting the data at each node by finding the best attribute and threshold to split at. As mentioned previously, Gini Index was used to determine how pure the dataset was, finding the best attribute and threshold that minimised the Gini value would indicate higher purity of the sample and thus would be the best split.

Based on the split, the data was split recursively into subsets. This process would continue until meeting the stopping criteria of the "max_depth" and "sample_min_split" or until the data in the node is completely pure. Once, reaching this criteria, the last data samples will become leaf nodes. The predicted class at these leaf node, if the decision tree met the stopping criteria, is the majority class in the leaf node samples.


### Sklearn DecisionTreeClassifier

A library Decision Tree implementation was used for the comparison against the custom implementation. The "DecisionTreeClassifier" was taken from the Sci-kit learn website and was imported into the Pycharm to be trained and tested. Similar to the custom decision tree, it was initialised with "max_depth", "min_sample_split" ("sample_min_split"- in the custom decision tree) hyper-parameters. 

In order to train the tree, the .fit() function was used.

Once, importing the datasets, the same testing sizes were used to maintain consistent testing of performance measurements across both decision tree implementations. Sklearn.metric imported the performance metrics used for measuring accuracy, precision, f1 score and recall.


### Computational Metrics

As mentioned previously, prediction time, training time and memory usage were measured and compared across both decision trees.

Training time was measured using the time.time() function in python which allowed to measure the prediction time through subtracting the start time from the final time, which was the time taken to finish the operation recorded in seconds.

To determine memory usage for both decision trees, memory_usage from memory_profiler was imported. However, once trialed, memory_usage was not recorded in the results. Thus, an alternative approach was used, as a fail-safe for when memory_profiler does not work. Python's tracemalloc() would take snapshots of before and after performing an operation, to compare how much memory it used. 

# 4. Results

## 4.1 Average Results

This section shows the average machine learning (accuracy, precision, recall and f1 score) and computational (training time, prediction time and memory usage ) performance of the custom and sklearn decision trees for the three datasets that they were tested on. 


### Machine Learning 

#### Small Dataset- Iris

Table 1 shows average scores (mean and SD) showing high machine learning metrics scores when tested in the Iris dataset. However, slightly higher scores from the sklearn decision tree showed better performance in machine learning aspects compared to the custom implementation. 

```{r} 
iris <- read_csv("Assignment/results_custom_vs_sklearn_dataset_53.csv")


iris_summary <- iris %>% # mean and sd for custom metrics
  summarise(mean_custom_acc = mean(custom_accuracy), 
            sd_custom_acc = sd(custom_accuracy), 
            mean_custom_recall = mean(custom_recall), 
            sd_custom_recall = sd(custom_recall), 
            mean_custom_precision = mean(custom_precision), 
            sd_custom_precision = sd(custom_precision), 
            mean_custom_f1 = mean(custom_f1), 
            sd_custom_f1 = sd(custom_f1)) 

# for a structured table with metrics at the side and columns for mean and sd:
# converting to long format first
iris_long <- iris_summary %>%
  pivot_longer(cols = everything(), 
               names_sep = "_custom_",
               names_to = c("stat", "metric"),
               values_to = "value")

# then pivot_wider()

iris_stats <- iris_long %>%
  pivot_wider(names_from = stat, 
              values_from = value)


# doing the same with the sklearn 
iris_summary_skl <- iris %>% # mean and sd for custom metrics
  summarise(mean_sklearn_acc = mean(sklearn_accuracy), 
            sd_sklearn_acc = sd(sklearn_accuracy), 
            mean_sklearn_recall = mean(sklearn_recall), 
            sd_sklearn_recall = sd(sklearn_recall), 
            mean_sklearn_precision = mean(sklearn_precision), 
            sd_sklearn_precision = sd(sklearn_precision), 
            mean_sklearn_f1 = mean(sklearn_f1), 
            sd_sklearn_f1 = sd(sklearn_f1)) 

iris_long_skl <- iris_summary_skl %>%
  pivot_longer(cols = everything(), 
               names_sep = "_sklearn_",
               names_to = c("stat", "metric"),
               values_to = "value")

# then pivot_wider()

iris_stats_skl <- iris_long_skl %>%
  pivot_wider(names_from = stat, 
              values_from = value)

combine_iris <- iris_stats %>% # combine the dataframes together
  rename(mean_custom = mean, sd_custom = sd) %>%
  left_join(iris_stats_skl %>%
              rename(mean_sklearn = mean, sd_sklearn = sd))

knitr::kable(combine_iris,  # turning it into a table with a caption
             caption = "Table 1. Iris Data :Mean and SD for Machine Learning Metrics")
```


#### Medium Dataset- Wine

The results for the medium data (Table 2) set showed similar results to the small Iris dataset. Although both decision tree implementations showed high results again with the medium-size Wine dataset, the average machine learning metrics from the custom implementation was found to be smaller than those of the sklearn decision tree. It was also noted the scores were lower than they were when tested against the small dataset.

```{r}

wine <- read_csv("Assignment/results_custom_vs_sklearn_dataset_109.csv")


wine_summary <- wine %>% # mean and sd for custom metrics
  summarise(mean_custom_acc = mean(custom_accuracy), 
            sd_custom_acc = sd(custom_accuracy), 
            mean_custom_recall = mean(custom_recall), 
            sd_custom_recall = sd(custom_recall), 
            mean_custom_precision = mean(custom_precision), 
            sd_custom_precision = sd(custom_precision), 
            mean_custom_f1 = mean(custom_f1), 
            sd_custom_f1 = sd(custom_f1)) 


# for a structured table with metrics at the side and columns for mean and sd:
# converting to long format first
wine_long <- wine_summary %>%
  pivot_longer(cols = everything(), 
               names_sep = "_custom_",
               names_to = c("stat", "metric"),
               values_to = "value")

# then pivot_wider()

wine_stats <- wine_long %>%
  pivot_wider(names_from = stat, 
              values_from = value)


# doing the same with the sklearn 
wine_summary_skl <- wine %>% # mean and sd for custom metrics
  summarise(mean_sklearn_acc = mean(sklearn_accuracy), 
            sd_sklearn_acc = sd(sklearn_accuracy), 
            mean_sklearn_recall = mean(sklearn_recall), 
            sd_sklearn_recall = sd(sklearn_recall), 
            mean_sklearn_precision = mean(sklearn_precision), 
            sd_sklearn_precision = sd(sklearn_precision), 
            mean_sklearn_f1 = mean(sklearn_f1), 
            sd_sklearn_f1 = sd(sklearn_f1)) 


wine_long_skl <- wine_summary_skl %>%
  pivot_longer(cols = everything(), 
               names_sep = "_sklearn_",
               names_to = c("stat", "metric"),
               values_to = "value")

# then pivot_wider()

wine_stats_skl <- wine_long_skl %>%
  pivot_wider(names_from = stat, 
              values_from = value)

combine_wine <- wine_stats %>% # combine the dataframes
  rename(mean_custom = mean, sd_custom = sd) %>%
  left_join(wine_stats_skl %>%
              rename(mean_sklearn = mean, sd_sklearn = sd))




knitr::kable(combine_wine,  # turning it into a table with a caption
             caption = "Table 2. Wine Data: Mean and SD for Machine Learning Metrics")
```

#### Large Dataset- Breast Cancer

The results have differed slightly compared to the two datasets, as seen in Table 3. Contrasting to the previous datasets, the machine learning performance metrics for the custom decision tree are shown to be greater than those of the sklearn decision tree. Although there was not a big difference, there is a slight difference in results, showing the custom decision tree performed better in terms of accuracy, precision, recall and f1 score.

```{r}
breast_cancer <- read_csv("Assignment/results_custom_vs_sklearn_dataset_17.csv")

bc_summary <- breast_cancer %>% # mean and sd for custom metrics
  summarise(mean_custom_acc = mean(custom_accuracy), 
            sd_custom_acc = sd(custom_accuracy), 
            mean_custom_recall = mean(custom_recall), 
            sd_custom_recall = sd(custom_recall), 
            mean_custom_precision = mean(custom_precision), 
            sd_custom_precision = sd(custom_precision), 
            mean_custom_f1 = mean(custom_f1), 
            sd_custom_f1 = sd(custom_f1)) 


# for a structured table with metrics at the side and columns for mean and sd:
# converting to long format first
bc_long <- bc_summary %>%
  pivot_longer(cols = everything(), 
               names_sep = "_custom_",
               names_to = c("stat", "metric"),
               values_to = "value")

# then pivot_wider()

bc_stats <- bc_long %>%
  pivot_wider(names_from = stat, 
              values_from = value)

# doing the same with the sklearn 
bc_summary_skl <- breast_cancer %>% # mean and sd for custom metrics
  summarise(mean_sklearn_acc = mean(sklearn_accuracy), 
            sd_sklearn_acc = sd(sklearn_accuracy), 
            mean_sklearn_recall = mean(sklearn_recall), 
            sd_sklearn_recall = sd(sklearn_recall), 
            mean_sklearn_precision = mean(sklearn_precision), 
            sd_sklearn_precision = sd(sklearn_precision), 
            mean_sklearn_f1 = mean(sklearn_f1), 
            sd_sklearn_f1 = sd(sklearn_f1)) 


bc_long_skl <- bc_summary_skl %>%
  pivot_longer(cols = everything(), 
               names_sep = "_sklearn_",
               names_to = c("stat", "metric"),
               values_to = "value")

# then pivot_wider()
bc_stats_skl <- bc_long_skl %>%
  pivot_wider(names_from = stat, 
              values_from = value)


combine_bc <- bc_stats %>% # combine the dataframes together
  rename(mean_custom = mean, sd_custom = sd) %>%
  left_join(bc_stats_skl %>%
              rename(mean_sklearn = mean, sd_sklearn = sd))

knitr::kable(combine_bc, # turning it into a table with a caption
             caption = "Table 3. Breast Cancer Data: Mean and SD for Machine Learning Metrics")

```
### Computational Aspects 




#### Train time 


##### Small dataset (Iris)
When determining the mean train time (in seconds) for both the custom and sklearn decision trees tested in the breast cancer dataset. Overall the findings showed a faster training time (in seconds) for the sklearn decision tree (M = 0.0014, SD = 0.00047) than the custom decision tree (M = 0.025, SD = 0.0095).




##### Medium dataset (Wine)

After determining the mean train time for the custom and sklearn implementations after being tested on the wine dataset, the results showed similar results to those in the other datasets, sklearn had a faster training time (M = 0.014, SD = 0.0005) compared to the custom decision tree (M = 0.215, SD = 0.114).



##### Large dataset (Breast Cancer)

The findings showed that train time was higher for the custom decision tree (M = 4.19, SD = 2.96) than the sklearn decision tree (M = 0.004, SD = 0.008). Therefore, showing that the sklearn decision tree was much faster to train. 





#### Prediction Time 


##### Small dataset (Iris)

As a small dataset was used to test for prediction time (in seconds) to compare both implementations of decision trees, the average prediction times for both are small. However, the results still showed a slightly faster prediction time for the sklearn implementation (M = 0.0000623, SD = 0.000244) compared to the custom implementation (M = 0.0000653, SD = 0.000243). 



##### Medium dataset (Wine)

After determining the mean preduction time for the custom and sklearn implementations after being tested on the wine dataset, the results showed similar results to those in the other datasets, sklearn had a faster training time (M = 0.000025, SD = 0.00016) compared to the custom decision tree (M = 0.000087, SD = 0.00028).



##### Large dataset (Breast Cancer)

When measuring the average prediction time for both decision tree implementations, even though the prediction time for both were low, the sklearn decision tree had a lower prediction time (M = 0.0001, SD = 0.003) and was therefore faster at predicting class labels than the custom decision tree (M = 0.0003, SD = 0.0005).




#### Memory Usage



##### Small dataset (Iris)

To determine which decision tree was more efficient, the average memory usage was measured to compare how much memory was used up (in MegaBytes) between each of the decision trees when testing the smallest data (Iris). Overall the findings showed a lower memory usage for the sklearn decision tree (M = 0.00116, SD = 0.00123) than the custom decision tree (M = 0.00311, SD = 0.014).



##### Medium dataset (Wine)

Similar to the other datasets tested, when tested on the medium dataset (Wine), the sklearn decision tree used up less memory (M = 0.0011, SD = 0.00063) than the custom decision tree (M = 0.00214, SD = 0.00212).




##### Large dataset (Breast Cancer)

Overall, the results showed the sklearn decision tree used up less memory (M = 0.0009, SD = 0.00000147) when classifying than the custom decision tree (M = 0.004, SD = 0.011) tested using the large datset. 




## 4.2 Graphs of Results

### 4.2.1 Accuracy and Max_Depth- Small Dataset

Visualisations for both decision tree implementations showed how Accuracy varies with Max_Depth in the Iris dataset

``` {r accuracy-max-depth-iris, fig.cap = "Iris: Boxplot for Accuracy and Max_Depth"}

# boxplot for custom accuracy vs max_depth
iris_custom_max_acc <- ggplot(iris, aes(x = as.factor(max_depth), y = custom_accuracy)) +
  geom_boxplot() +
  labs(x = "Max Depth", y = "Custom Accuracy")

# creating boxplots for sklearn accuracy vs max_depth
iris_sklearn_max_acc <- ggplot(iris, aes(x = as.factor(max_depth), y = sklearn_accuracy)) +
  geom_boxplot() +
  labs(x = "Max Depth", y = "Sklearn Accuracy")

# putting the plots next to each other
iris_custom_max_acc | iris_sklearn_max_acc 

# using the patchworks library
```

This figure shows how accuracy differs depending on varied maximum levels of tree depth. Custom decision tree implementation shows distribution of greater accuracy levels at max_depth of 3 similarly to the sklearn decision tree, and showed the lowest accuracy level at max_depth of 1. 



### 4.2.2 Accuracy and Max_Depth- Medium Dataset

Visualisation for accuracy and the hyper-parameter of max_depth in the Wine dataset.

```{r fig.cap = "Wine: Boxplot for Accuracy and Max_Depth"}


# boxplot for custom accuracy and max_depth
wine_custom_max_acc <- ggplot(wine, aes(x = as.factor(max_depth), y = custom_accuracy)) +
  geom_boxplot() +
  labs(x = "Max Depth", y = "Custom Accuracy")


# boxplot for sklearn accuracy and max_depth
wine_sklearn_max_acc <- ggplot(wine, aes(x = as.factor(max_depth), y = sklearn_accuracy)) +
  geom_boxplot() +
  labs(x = "Max Depth", y = "Sklearn Accuracy")


# putting the boxplots next to each other
wine_custom_max_acc | wine_sklearn_max_acc

# using patchworks library
```
This figure shows overall higher distribution of sklearn accuracy scores across all levels of max_depth compared to custom decision tree custom score. Excluding maximum depth of 1 where both implementations had similar distribution of low accuracy scores.


### 4.2.3 Accuracy and Max_Depth- Large Dataset

Visualisation for accuracy and the hyper-parameter of max_depth in the Breast Cancer dataset.

```{r fig.cap = "Breast Cancer: Boxplot for Accuracy and Max_Depth"}


# boxplot for custom accuracy and max_depth
bc_custom_max_acc <- ggplot(breast_cancer, aes(x = as.factor(max_depth), y = custom_accuracy)) +
  geom_boxplot() +
  labs(x = "Max Depth", y = "Custom Accuracy")


# boxplot for sklearn accuracy and max_depth
bc_sklearn_max_acc <- ggplot(breast_cancer, aes(x = as.factor(max_depth), y = sklearn_accuracy)) +
  geom_boxplot() +
  labs(x = "Max Depth", y = "Sklearn Accuracy")


# putting the boxplots next to each other
bc_custom_max_acc | bc_sklearn_max_acc

# using patchworks library

```
This figure shows bigger distribution of high accuracy scores for the sklearn decision tree at maximum depth of 2 and 3. However, there is a bigger distribution for lower accuracy scores at maximum depth of 5 when compared against the custom decision tree. 


Therefore these results show the effect of tree depth on accuracy for both decision trees suggesting that a decision tree's maximum depth should not be too high or too low. 



# 4.3 Statistical Analyses

Paired T-tests were conducted to compare machine learning and computational aspects of both decision tree implementations. 



## Computational T-tests

A paired t-test comparing training times between the two decision tree implementations showed there was a statistically significant difference (t(79)= 16.79, p < 2.2e-16) showing the custom decision tree was 0.21 seconds longer.

When comparing memory usage between the two trees, there was a statistically significant difference (t(79)= 4.61, p = 0.000015) where there was a mean difference of 0.0011 MB.

However, when comparing the prediction times, there no statistical significant difference showing similar prediction times between the two implementations.




## Machine Learning T-tests



Two-Way Analyses of variance (ANOVAs) were conducted to compare machine learning aspects against the hyper-parameters of "max_depth" and "sample_min_split". The results are seen in Table 4 for the Iris dataset.

```{r}

# anovas for comparing performance metrics with hyper-parameters

# anova for custom accuracy 
iris_custom_acc_anova <- aov(custom_accuracy ~ max_depth * sample_min_split, data = iris)

# anova for custom recall
iris_custom_rec_anova <- aov(custom_recall ~ max_depth * sample_min_split, data = iris)

# anova for custom precision
iris_custom_precision_anova <- aov(custom_precision ~ max_depth * sample_min_split, data = iris)

# anova for custom f1 score
iris_custom_f1_anova <- aov(custom_f1 ~ max_depth * sample_min_split, data = iris)


# anova for sklearn accuracy
iris_sklearn_acc_anova <- aov(sklearn_accuracy ~ max_depth * sample_min_split, data = iris)

# anova for sklearn recall
iris_sklearn_rec_anova <- aov(sklearn_recall ~ max_depth * sample_min_split, data = iris)

# anova for sklearn precision
iris_sklearn_precision_anova <- aov(sklearn_precision ~ max_depth * sample_min_split, data = iris)

# anova for sklearn f1 score
iris_sklearn_f1_anova <- aov(sklearn_f1 ~ max_depth * sample_min_split, data = iris)

# using tidy() function from "broom" library to tidy and bind all anova results  
iris_anova_results <- bind_rows(tidy(iris_custom_acc_anova) %>% mutate(metric = "Custom Accuracy"), 
                           tidy(iris_custom_rec_anova) %>% mutate(metric = "Custom Recall"), 
                           tidy(iris_custom_precision_anova) %>% mutate(metric = "Custom Precision"),
                           tidy(iris_custom_f1_anova) %>% mutate(metric = "Custom f1 score"), 
                           tidy(iris_sklearn_acc_anova) %>% mutate(metric = "Sklearn Accuracy"), 
                           tidy(iris_sklearn_rec_anova) %>% mutate(metric = "Sklearn Recall"), 
                           tidy(iris_sklearn_f1_anova) %>% mutate(metric = "Sklearn f1 score"))
# using bind_rows() to bind the rows together from the "dlpyr" library

iris_anova_results

# selecting specific columns
iris_anova_table <- iris_anova_results %>% select("metric", "p.value", "term", "df", "statistic")

iris_anova_table$p.value <- as.character(iris_anova_table$p.value) # turning "p.value" into a character
iris_anova_table$p.value <- replace_na(iris_anova_table$p.value, "-") # to be able to add "-" to NA values
iris_anova_table$statistic <- as.character(iris_anova_table$statistic)  # turning "p.value" into a character
iris_anova_table$statistic <- replace_na(iris_anova_table$statistic, "-") 

knitr::kable(iris_anova_table, 
             caption = "Table 4. Anova Table for hyperparameters with machine learning aspects with Iris data")


```
These results showed "max_depth" having a statistically significant effect on machine learning aspects in the Iris dataset.

```{r}

# anovas for comparing performance metrics with hyper-parameters

# anova for custom accuracy 
wine_custom_acc_anova <- aov(custom_accuracy ~ max_depth * sample_min_split, data = wine)

# anova for custom recall
wine_custom_rec_anova <- aov(custom_recall ~ max_depth * sample_min_split, data = wine)

# anova for custom precision
wine_custom_precision_anova <- aov(custom_precision ~ max_depth * sample_min_split, data = wine)

# anova for custom f1 score
wine_custom_f1_anova <- aov(custom_f1 ~ max_depth * sample_min_split, data = wine)

# anova for sklearn accuracy
wine_sklearn_acc_anova <- aov(sklearn_accuracy ~ max_depth * sample_min_split, data = wine)

# anova for sklearn recall
wine_sklearn_rec_anova <- aov(sklearn_recall ~ max_depth * sample_min_split, data = wine)

# anova for sklearn precision
wine_sklearn_precision_anova <- aov(sklearn_precision ~ max_depth * sample_min_split, data = wine)

# anova for sklearn f1 score
wine_sklearn_f1_anova <- aov(sklearn_f1 ~ max_depth * sample_min_split, data = wine)

# using tidy() function from "broom" library to tidy and bind all anova results  
wine_anova_results <- bind_rows(tidy(wine_custom_acc_anova) %>% mutate(metric = "Custom Accuracy"), 
                           tidy(wine_custom_rec_anova) %>% mutate(metric = "Custom Recall"), 
                           tidy(wine_custom_precision_anova) %>% mutate(metric = "Custom Precision"),
                           tidy(wine_custom_f1_anova) %>% mutate(metric = "Custom f1 score"), 
                           tidy(wine_sklearn_acc_anova) %>% mutate(metric = "Sklearn Accuracy"), 
                           tidy(wine_sklearn_precision_anova) %>% mutate(metric = "Sklearn Precision"), 
                           tidy(wine_sklearn_rec_anova) %>% mutate(metric = "Sklearn Recall"), 
                           tidy(wine_sklearn_f1_anova) %>% mutate(metric = "Sklearn f1 score"))

# selecting specific columns
wine_anova_table <- wine_anova_results %>% select("metric", "p.value", "term", "df", "statistic")

wine_anova_table$p.value <- as.character(wine_anova_table$p.value) # turning "p.value" into a character
wine_anova_table$p.value <- replace_na(wine_anova_table$p.value, "-") # to be able to add "-" to NA values
wine_anova_table$statistic <- as.character(wine_anova_table$statistic)  # turning "p.value" into a character
wine_anova_table$statistic <- replace_na(wine_anova_table$statistic, "-") 

knitr::kable(wine_anova_table, # turning it into a table with a caption
      caption = "Table 5. Anova Table for hyperparameters with machine learning aspects with Wine data")

```

These results show that the hyper-parameter of maximum tree depth had an statistically significant effect on accuracy, recall, precision and f1 score for both decision tree implementations.



```{r}
# anovas for comparing performance metrics with hyper-parameters

# anova for custom accuracy 
bc_custom_acc_anova <- aov(custom_accuracy ~ max_depth * sample_min_split, data = breast_cancer)

# anova for custom recall
bc_custom_rec_anova <- aov(custom_recall ~ max_depth * sample_min_split, data = breast_cancer)

# anova for custom precision
bc_custom_precision_anova <- aov(custom_precision ~ max_depth * sample_min_split, data = breast_cancer)

# anova for custom f1 score
bc_custom_f1_anova <- aov(custom_f1 ~ max_depth * sample_min_split, data = breast_cancer)

# anova for sklearn accuracy
bc_sklearn_acc_anova <- aov(sklearn_accuracy ~ max_depth * sample_min_split, data = breast_cancer)

# anova for sklearn recall
bc_sklearn_rec_anova <- aov(sklearn_recall ~ max_depth * sample_min_split, data = breast_cancer)

# anova for sklearn precision
bc_sklearn_precision_anova <- aov(sklearn_precision ~ max_depth * sample_min_split, data = breast_cancer)

# anova for sklearn f1 score
bc_sklearn_f1_anova <- aov(sklearn_f1 ~ max_depth * sample_min_split, data = breast_cancer)

# using tidy() function from "broom" library to tidy and bind all anova results  
bc_anova_results <- bind_rows(tidy(bc_custom_acc_anova) %>%  mutate(metric = "Custom Accuracy"),
                           tidy(bc_custom_rec_anova) %>% mutate(metric = "Custom Recall"), 
                           tidy(bc_custom_precision_anova) %>%  mutate(metric = "Custom Precision"),
                           tidy(bc_custom_f1_anova) %>% mutate(metric = "Custom f1 score"), 
                           tidy(bc_sklearn_acc_anova) %>% mutate(metric = "Sklearn Accuracy"), 
                           tidy(bc_sklearn_precision_anova) %>% mutate(metric = "Sklearn Precision"), 
                           tidy(bc_sklearn_rec_anova) %>% mutate(metric = "Sklearn Recall"), 
                           tidy(bc_sklearn_f1_anova) %>% mutate(metric = "Sklearn f1 score"))


# selecting specific columns
bc_anova_table <- bc_anova_results %>% select("metric", "p.value", "term", "df", "statistic")

bc_anova_table$p.value <- as.character(bc_anova_table$p.value) # turning "p.value" into a character
bc_anova_table$p.value <- replace_na(bc_anova_table$p.value, "-") # to be able to add "-" to NA values
bc_anova_table$statistic <- as.character(bc_anova_table$statistic)  # turning "p.value" into a character
bc_anova_table$statistic <- replace_na(bc_anova_table$statistic, "-") 

knitr::kable(bc_anova_table, 
             caption = "Table 6. Anova Table for hyperparameters with machine learning aspects with Breast Cancer data")


```
However, in the breast cancer dataset "max_depth" hyperparameter only had a statistically significant effect on recall and precision not accuracy and f1 score for both decision tree implementations.

# 5. Discussion



## 5.1 Machine Learning Performance

The Sklearn Decision Implementation worked well through showing higher results of Machine Learning Performance scores; showing higher accuracy, precision, recall and f1 when compared against the custom decision tree and tested across the same performance measures. 



## 5.2 Computational Performance

As previously mentioned, the sklearn implementation performed well showing faster testing times compared to the custom implementation as well as using up less memory. However, after a paired t-test comparing predicted time there was no statistically significant difference of prediction time, therefore suggesting both decision tree implementations were close in prediction time. 



# 6. Conclusion

In conclusion, this project aimed to evaluate the machine learning and computational aspects of a sklearn library decision tree and a custom implemented decision tree to investigate their effectiveness. 

The analysis showed that overall the sklearn decision implementation was more efficient scoring highly in machine learning (accuracy, precision, recall and f1 score) as well as computational (training time, prediction time and memory usage) aspects. However, it was shown that although there were a lot of statistically significant results, the sklearn prediction time was closely similar to the custom prediction time, therefore was deemed statistically insignificant. 

Hyper-parameters were also measured to see their effect on machine learning aspects for both implementations. Although it was shown that there was no statistical effect with minimum sample split and its interaction with maximum depth, when just looking at maximum tree depth alone there was a significant effect on the machine learning aspects for both decision tree implementations.



# 7. Acknowledgements

I had some help from peers such as friends who take computer science and take a module on Artificial Intelligence, NumPy was used to help with coding the custom decision tree, Pandas was also used for converting results to .csv() files. For the R code the libraries used were: patchworks, broom, tidyverse, and ggplot2. 

# References

1.10. Decision Trees. (n.d.). Scikit-learn. https://scikit-learn.org/stable/modules/tree.html#classification 

Aaboub, F., Chamlal, H., & Ouaderhman, T. (2023). Statistical analysis of various splitting criteria for decision trees. Journal of Algorithms & Computational Technology, 17. https://doi.org/10.1177/17483026231198181 

Decision Tree Sklearn -Depth Of tree and accuracy. (n.d.). Stack Overflow. https://stackoverflow.com/questions/49289187/decision-tree-sklearn-depth-of-tree-and-accuracy 

GeeksforGeeks. (2024, October 11). ML | Gini Impurity and Entropy in Decision Tree. GeeksforGeeks. https://www.geeksforgeeks.org/gini-impurity-and-entropy-in-decision-tree-ml/ 

Importance of decision tree hyperparameters on generalization — Scikit-learn course. (n.d.). https://inria.github.io/scikit-learn-mooc/python_scripts/trees_hyperparameters.html#:~:text=The%20hyperparameter%20max_depth%20controls%20the,the%20impact%20of%20the%20parameter. 

Kaushik, A. (2023, June 18). Gini Impurity and entropy for decision tree - Arpita Kaushik - medium. Medium. https://medium.com/@arpita.k20/gini-impurity-and-entropy-for-decision-tree-68eb139274d1 

UCI Machine Learning Repository. (n.d.). https://archive.ics.uci.edu/ 

What is sklearn? | Domino Data Lab. (n.d.). Domino Data Lab. https://domino.ai/data-science-dictionary/sklearn
