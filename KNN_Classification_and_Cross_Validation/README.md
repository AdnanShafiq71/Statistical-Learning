# KNN Classification and Cross-Validation

## Overview

This project investigates the use of **K-Nearest Neighbours (KNN)** for binary classification using a diabetes dataset.

The analysis focuses on how the choice of the hyperparameter \(K\) affects predictive performance and demonstrates how **stratified cross-validation** can be used to select an appropriate value of \(K\).

The complete modelling workflow is implemented in a single Python script, `KNN_model.py`.

---

## Objectives

The main objectives of this project are to:

* Explore and assess the quality of the dataset.
* Investigate the distribution of the target variable.
* Apply appropriate preprocessing for a distance-based classification model.
* Implement K-Nearest Neighbours classification.
* Use stratified 5-fold cross-validation to evaluate different values of \(K\).
* Select the value of \(K\) with the highest mean cross-validation accuracy.
* Evaluate the selected model on an unseen test set.
* Analyse model performance using multiple classification metrics and a confusion matrix.

---

## Dataset

The dataset contains **1,879 observations** and **46 original variables**.

The target variable is:

* `Diagnosis = 0` — No Diabetes
* `Diagnosis = 1` — Diabetes

Two identifier variables, `PatientID` and `DoctorInCharge`, were removed because they are not used as predictive features.

After removing these variables, the remaining predictors are numerical and therefore suitable for standardisation and KNN classification.

### Class Distribution

The dataset contains:

| Diagnosis   | Number of observations |
| ----------- | ---------------------: |
| No Diabetes |                  1,127 |
| Diabetes    |                    752 |

The class distribution is therefore somewhat imbalanced, making it useful to consider metrics beyond accuracy when evaluating the final model.

---

## Methodology

### 1. Data Inspection and Quality Checks

The dataset was examined for:

* Number of observations and variables
* Variable names and data types
* Summary statistics
* Missing values
* Duplicate observations
* Potential outliers
* Target-class distribution

No missing values or duplicate observations were identified.

Potential outliers were investigated using the **Interquartile Range (IQR)** method.

---

### 2. Train/Test Split

The data was divided into:

* **80% training data**
* **20% test data**

A fixed random seed of `42` was used to ensure reproducibility.

Stratification was applied to preserve the class distribution between the training and test sets.

---

### 3. Standardisation

KNN is a distance-based algorithm, meaning that differences in feature scales can influence the calculated distances between observations.

The predictors were therefore standardised using `StandardScaler`.

Standardisation was incorporated into a **scikit-learn Pipeline** together with the KNN classifier.

This ensures that scaling is fitted only on the relevant training data during cross-validation, preventing information from the validation folds from leaking into the preprocessing step.

---

### 4. K-Nearest Neighbours

KNN classifies an observation based on the classes of its nearest neighbours.

The number of neighbours, \(K\), was treated as the main hyperparameter.

Values of:

$$
K = 1,2,\ldots,30
$$

were evaluated.

---

### 5. Cross-Validation

A **5-fold Stratified Cross-Validation** procedure was used on the training data.

For each value of \(K\):

1. The training data was divided into five folds.
2. Four folds were used to train the model.
3. The remaining fold was used for validation.
4. This process was repeated until each fold had been used as the validation set.
5. The mean validation accuracy was calculated.

The value of \(K\) producing the highest mean cross-validation accuracy was selected as the final hyperparameter.

The test set was not used during this selection process.

---

## Model Evaluation

After selecting the optimal \(K\), the final KNN model was fitted using the complete training dataset.

Performance was then evaluated on the previously unseen test set using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **Confusion Matrix**

A classification report is also generated and saved as a CSV file.

A majority-class baseline is considered to provide context for the model's accuracy.

---

## Visualisations

The project generates several outputs to support the analysis.

### Summary Statistics

`summary_statistics.png`

Provides descriptive statistics for the numerical variables in the dataset.

### Diagnosis Distribution

`diagnosis_distribution.png`

Shows the distribution of diabetic and non-diabetic observations.

### KNN Performance Across K

`knn_k_vs_cv_accuracy.png`

Shows the relationship between the number of neighbours and mean 5-fold cross-validation accuracy.

### Top K Values

`top_5_knn_results.png`

Displays the five values of \(K\) producing the highest mean cross-validation accuracy.

### Final Model Performance

`final_model_performance.png`

Summarises the final model's accuracy, precision, recall and F1 score.

### Confusion Matrix

`knn_confusion_matrix.png`

Shows the number of correct and incorrect predictions made by the final model for each class.

---

## Repository Structure

```text
KNN_Classification_and_Cross_Validation/
│
├── data/
│   ├── diabetes_data.csv
│   └── classification_report.csv
│
├── plots/
│   ├── summary_statistics.png
│   ├── diagnosis_distribution.png
│   ├── knn_k_vs_cv_accuracy.png
│   ├── top_5_knn_results.png
│   ├── final_model_performance.png
│   └── knn_confusion_matrix.png
│
└── src/
    └── KNN_model.py
```

---

## Technologies

* **Python**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**

Key scikit-learn components used include:

* `train_test_split`
* `StratifiedKFold`
* `cross_val_score`
* `StandardScaler`
* `Pipeline`
* `KNeighborsClassifier`
* `accuracy_score`
* `precision_score`
* `recall_score`
* `f1_score`
* `confusion_matrix`

---

## Key Statistical Learning Concepts

This project demonstrates practical applications of:

* Supervised learning
* Binary classification
* Distance-based learning
* Feature standardisation
* Train/test splitting
* Cross-validation
* Hyperparameter tuning
* Model selection
* Classification metrics
* Bias-variance considerations
* Data leakage prevention
* Model error analysis

---

## Reproducibility

A fixed `random_state=42` is used for the train/test split and cross-validation procedure.

The modelling pipeline also ensures that standardisation is performed correctly within the cross-validation process.

The analysis can therefore be reproduced by running:

```bash
python src/KNN_model.py
```

---

## Limitations

KNN has several limitations that should be considered when interpreting the results:

* Its predictions depend on the choice of distance metric and \(K\).
* Performance can be sensitive to feature scaling.
* KNN can become computationally expensive as the number of observations increases.
* The choice of accuracy as the optimisation metric may not fully capture performance when class distributions are imbalanced.
* The model identifies patterns based on similarity between observations but does not provide a straightforward parametric interpretation of individual predictors.

---

## Future Extensions

Possible extensions to the project include:

* Comparing KNN with logistic regression and other classification methods.
* Investigating alternative distance metrics.
* Evaluating different weighting schemes for neighbouring observations.
* Comparing additional performance metrics such as ROC-AUC.
* Performing more detailed error analysis.
* Investigating feature selection or dimensionality reduction techniques such as PCA.
