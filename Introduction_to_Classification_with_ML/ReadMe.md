## Introduction

# Dataset
A dataset was obtained by preparing a 12-question test. This test is related to self-confidence level.
After individuals answered the questions, they labeled the data by answering the last question, **"Finally, on a scale of 1-5, rate your self-confidence."** In the task, self-confidence classification is conducted using the other 11 questions.

# Data Preprocessing
Except for the last question, each question in the test has 3 different options. These options are categorical data. **One Hot Encoding** method is used to convert categorical data into numerical data.
One Hot Encoding transforms each distinct value of a categorical variable into an array. This array consists of zeros for all possible values of the variable, except for the selected value, which is represented by a single 1. In other words, each category is represented as a separate column.
For example, let's consider a "color" variable with possible values of "red," "blue," and "green." When One Hot Encoding is applied, the "color" variable is represented as [1, 0, 0] for "red," [0, 1, 0] for "blue," and [0, 0, 1] for "green."
The purpose of One Hot Encoding is to convert categorical data into a numerical format, enabling machine learning algorithms to work more effectively. This method emphasizes that there is no linear relationship between category values, allowing algorithms to interpret the data more accurately.
One Hot Encoding is a widely used technique, especially in machine learning models, when processing categorical variables is required.

# Classification with Machine Learning
Five machine learning algorithms have been selected for classification. These algorithms are **Logistic Regression (LR)**, **K-Nearest Neighbors (KNN)**, **Decision Tree (DT)**, **Support Vector Classifier (SVC)**, and **Random Forest (RF)**.

Typically, datasets are divided into 70% training and 30% testing, or 75% training and 25% testing. However, due to the limited amount of data collected in this study (200 rows), the dataset has been split into 90% training and 10% testing.

**Note:** The decision boundaries of the following algorithms with respect to the given data in the assignment are based on the data after applying PCA (Principal Component Analysis). PCA will be explained in the later parts of the assignment.

# Logistic Regression (LR)
Logistic Regression is a statistical regression method used to evaluate and predict the relationship between a dependent variable and independent variables. It is commonly used in binary classification problems.
Logistic Regression works similarly to linear regression but has a different output. In linear regression, the dependent variable takes continuous values, while in Logistic Regression, it estimates the probability of the dependent variable. This probability value is between 0 and 1 and usually represents the likelihood of belonging to a class (label).
Logistic Regression uses the sigmoid (logistic) function to calculate the probability value of the dependent variable. This function transforms any real number into a value between 0 and 1.
The training of a Logistic Regression model often involves optimization methods such as maximum likelihood or gradient descent to find the optimal weight values. Once training is completed, the model can make predictions on new data and assign an observation to a specific class.
Logistic Regression is commonly used in classification problems and has many applications. For example, it can be used in marketing to predict customer churn, in medical fields for disease diagnosis, and in credit risk assessment, among others.

![lr](https://github.com/kursatkomurcu/Machine-Learning/blob/main/Introduction_to_Classification_with_ML/lr_decision_boundry.png)

# K-Nearest Neighbors (KNN)
K-Nearest Neighbors (KNN) is an algorithm used in the fields of machine learning and data mining to solve classification and regression problems. KNN primarily performs classification or prediction based on the neighbors of an instance.

When the KNN algorithm operates, it uses a metric to determine the neighbors of each instance in the dataset.

The algorithm compares each instance in the dataset with other instances based on this distance and finds the k nearest neighbors. The parameter "k" specifies the number of neighbors. When making a prediction, KNN works on the principle of majority class. It examines the classes of the k nearest neighbors and provides the majority class as the prediction. For example, when k is set to 3, it examines the classes of the closest 3 neighbors and makes a class prediction based on the majority.

KNN algorithm can also be used in regression problems. For regression, it takes the values of the k nearest neighbors and calculates their average or weighted average to form the predicted value.

KNN algorithm has some important characteristics:

**1.** KNN is a simple and understandable method.
**2.** It does not have a training process, the entire dataset is stored in memory.
**3.** The performance of KNN depends on parameters such as the number of neighbors and the distance measurement.
**4.** KNN algorithm can be sensitive to anomalies and noise in the dataset.
**5.** The computational cost of the KNN algorithm increases as the size of the dataset grows.

KNN is a widely used algorithm in classification and regression problems. However, in practice, it may have some limitations due to the computational cost in large datasets and high-dimensional data.

![knn](https://github.com/kursatkomurcu/Machine-Learning/blob/main/Introduction_to_Classification_with_ML/knn_decision_boundry.png)

# Decision Tree (DT)
Decision trees represent a tree-like structure used to make decisions based on attributes in a database.

The decision tree algorithm creates a series of decision rules based on the attributes in the dataset and classifies or predicts the data according to these rules. When decision trees perform classification or regression using attributes in the dataset, each internal node is split by an attribute test, and each branch corresponds to an attribute value.

The structure of decision trees is created by splitting the attributes and classes in the dataset based on information gain. Information gain measures the amount of information obtained by using an attribute. The goal is to divide the dataset in the best possible way at each step by selecting the attribute with the highest information gain.

When the decision tree algorithm is used for classification problems, each leaf node represents a class label. To make predictions, the input dataset follows the tree structure and reaches the relevant leaf node, and the class label of that node is provided as the prediction.

For regression problems, leaf nodes contain real numerical values. To make regression predictions, the input dataset follows the tree structure and reaches the relevant leaf node, and the value within that node is provided as the prediction.

Decision trees are easy to understand and interpret. They also have the ability to use categorical and numerical attributes together. Additionally, decision trees naturally handle multi-class problems and can handle missing data values.

However, decision trees have some weaknesses such as being prone to overfitting and being sensitive to small changes in the dataset. Moreover, they may encounter performance issues in very large and complex datasets.

![dt](https://github.com/kursatkomurcu/Machine-Learning/blob/main/Introduction_to_Classification_with_ML/dt_decision_boundry.png)

# Support Vector Classifier (SVC)
The support vector classifier utilizes support vectors to classify data points into different classes. The objective of the support vector classifier is to create a decision boundary between two classes. This decision boundary is defined as a hyperplane that best separates the classes. Support vectors are the closest points on this hyperplane and play a significant role in determining the decision boundary.

The support vector classifier can be used for linearly separable datasets, as well as for linearly inseparable datasets by utilizing various kernel functions. Kernel functions transform data points into a high-dimensional feature space, thereby making the dataset linearly separable.

To classify data points, the support vector classifier solves optimization problems to determine the support vectors and create the decision boundary. These optimization problems are based on the principle of maximum margin. The margin represents the distance between the decision boundary and the closest support vectors, and according to the principle of maximum margin, the decision boundary that best separates the classes maximizes the margin.

The support vector classifier can effectively work in high-dimensional feature spaces by using a technique called the "kernel trick." This allows it to separate linearly inseparable datasets with a linear hyperplane.

The support vector classifier generally exhibits good generalization performance as it is less prone to overfitting. It works effectively with smaller-sized datasets and is resistant to outliers.

The support vector classifier is a widely used algorithm in classification problems, especially in binary classification.

![svc](https://github.com/kursatkomurcu/Machine-Learning/blob/main/Introduction_to_Classification_with_ML/svc_decision_boundry.png)

# Random Forest (RF)
Random Forest is an ensemble learning method used for solving classification and regression problems in the field of machine learning. It combines multiple decision trees to create a stronger and more stable model.

The Random Forest algorithm works by creating multiple decision trees and aggregating their results. Each decision tree is trained using random samples (bootstrap samples) from the dataset. Bootstrap sampling involves randomly selecting samples from the dataset to create new subsets of data. This process allows each tree to be trained on different samples, introducing diversity.

During the construction of decision trees, at each node, a random subset of features is selected, and the tree is grown based on the best split criterion among these features. This feature selection process adds diversity among the trees and enables the overall model to make better generalizations.

In classification problems, the Random Forest model predicts the majority class by combining the class votes of the decision trees. In regression problems, the outputs of the trees are aggregated to obtain a prediction value such as the mean or median.

The Random Forest algorithm offers several advantages:

**1.** Random Forest is effective in handling high-dimensional datasets and working with multiple features.
**2.** It is resistant to overfitting and exhibits high generalization ability.
**3.** It provides successful results in both classification and regression problems.
**4.** It can handle missing values and outliers in the dataset.
**5.** Random Forest allows determining the importance ranking of features, identifying significant features in the dataset.

The Random Forest algorithm is widely used in classification and regression problems and has a track record of delivering successful results. Its high accuracy contributes to its popularity and broad usage.

![rf](https://github.com/kursatkomurcu/Machine-Learning/blob/main/Introduction_to_Classification_with_ML/rf_decision_boundry.png)

# Principal Component Analysis (PCA)
PCA (Principal Component Analysis) is a statistical technique used for data analysis and dimensionality reduction. It is employed to understand the patterns of variability within a dataset and represent the data with a reduced number of variables.

The primary objective of PCA is to summarize the relationships between variables in a multi-dimensional dataset. These relationships are determined based on the covariance matrix among the original variables in the dataset. By analyzing this covariance matrix, PCA obtains a new set of variables called "principal components" that describe the variability structures in the data.

PCA is utilized to reduce the dimensionality of a dataset while preserving as much information about the variability as possible. It reduces the dimensionality of the data, thereby reducing noise, eliminating unnecessary information, and providing a more easily interpretable structure for analysis.

The resulting principal components from PCA reflect the variability structure of the original dataset. The first principal component explains the maximum variability in the dataset, followed by subsequent principal components that account for the remaining variability. This way, a reduced-dimensional representation of the data can be obtained.

PCA is a widely used method in data analysis, pattern recognition, image processing, bioinformatics, and many other fields. It has various applications, including dimensionality reduction, data visualization, and noise reduction.

The table below shows the accuracy rates of the machine learning algorithms used in the assignment and their performance after applying PCA to the data.

![acc](https://github.com/kursatkomurcu/Machine-Learning/blob/main/Introduction_to_Classification_with_ML/acc.png)

# T-Test
T-test is a hypothesis test used to evaluate whether there is a statistically significant difference between the means of two sample groups. It is a statistical test employed to determine if the difference between two groups is likely due to chance.

The statistical value obtained from the T-test, known as the t-value, takes into account the magnitude of the difference between the group means and the sample size. This value is compared to critical t-values to determine if there is statistical significance.

T-test is a commonly used method in statistical analysis and plays a significant role in obtaining statistical results such as hypothesis tests, confidence intervals, and p-values.

The positive or negative outcomes of the T-test performed with machine learning models indicate a significant difference in performance between different models. However, it is important to consider certain factors to better understand the meaning of these results.

**Data used in the T-test:** The dataset used to compare the performance of different models is crucial. The dataset can vary in terms of factors such as the number of features, data distribution, and the nature of the target variable. Therefore, test results conducted on different datasets may vary.

**Type of T-test:** The type of T-test applied can also influence the results. It is important to choose the appropriate type of T-test to determine if there is a significant difference between models. This can include independent two-sample T-test or other types of T-tests.

**Directionality:** T-test results can be positive or negative, but this does not indicate the direction of which model performs better. A positive T-test result may suggest that one model performs better than the other, while a negative T-test result may indicate that the other model performs better. The direction of the results depends on the context of the test applied and the models being compared.

**Statistical significance:** The positive or negative outcome of the T-test alone does not provide meaningful information. It is important to evaluate the p-value to determine statistical significance. The p-value indicates whether the observed difference is likely due to chance. Low p-values indicate a significant difference between different models, while high p-values suggest that the difference is likely due to chance.

In conclusion, the positive or negative results of a T-test indicate a significant difference in performance between different models. However, it is important to consider other factors and evaluate statistical significance to fully understand the meaning of these results.

# Comment
Among the algorithms tested in the created dataset, Logistic Regression achieved the highest accuracy rate (80%), while Decision Tree had the lowest accuracy rate (45%). After applying PCA, the accuracy rates of the four algorithms slightly decreased, except for the Decision Tree algorithm, which had an increased accuracy rate.

By collecting more data, adding more questions to the test, or selecting different algorithms from these five, the accuracy rate can be further improved.
