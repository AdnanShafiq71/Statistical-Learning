import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


#Load the diabetes dataset and assigning dataframe to a variable called df
df = pd.read_csv("data/diabetes_data.csv")

#I want to check if the data has been loaded correctly, so I asked to print the first 10 rows of the dataset.
print(df.head(10))

#I want to check number of observations and variables in the dataset.
print(df.shape)
#There are 1879 observations and 46 variables in the dataset.

#I want to know the names of each of the variables in the dataset.
print(df.columns)
#From this, I can see that out of the 46 variables, 1 of them (Diagnosis) is my dependent variable and PatientID and DoctorInCharge are irrelevant to my analysis so I will drop them.
df = df.drop(columns=['PatientID', 'DoctorInCharge'])
#Check to see if the columns have been dropped successfully.
print(df.columns)

#I want to check the data types of each variable in the dataset.
print(df.dtypes)
#They are all either int64 or float64, as long as they are all numerical variables, it is fine.

#I want to do a summary statistic now
summary_stats = df.describe().T
print(summary_stats)
#I want to save the summary statistic as a table into my plots folder.
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("off")
table = ax.table(
    cellText=summary_stats.round(2).values,
    colLabels=summary_stats.columns,
    rowLabels=summary_stats.index,
    loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)
plt.savefig("plots/summary_statistics.png", bbox_inches="tight", dpi=300)
plt.close()

#------------DATA QUALITY CHECKS-----------------

#1. I want to check for missing values in the dataset.
print(df.isnull().sum())
#There are no missing values in the dataset, which is good.

#2. I want to check for duplicate rows in the dataset.
print(df.duplicated().sum())
#There are no duplicate rows in the dataset, which is good.

#3. I want to check for outliers in the dataset using boxplots.
#I want to check for how many outliers are present in each variable using the IQR method.
outliers_count = {}
for column in df.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outliers_count[column] = outliers.shape[0]

print(outliers_count)
#There is nothing alarming from our numerical variables that stops me from using KNN for my analysis.   


#I want to see the distribution of the dependent variable (Diagnosis) in the dataset.
diagnosis_counts = df['Diagnosis'].value_counts()
print(diagnosis_counts)
#There are 1127 non-diabetic patients and 752 diabetic patients in the dataset. I want to visualise this distribution using a bar chart.
plt.figure(figsize=(8, 6))
plt.bar(diagnosis_counts.index, diagnosis_counts.values, color=['blue', 'orange'])
plt.title('Distribution of Diagnosis')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.savefig("plots/diagnosis_distribution.png", bbox_inches="tight", dpi=300)
plt.close()




#I want to identify my predictors
y = df['Diagnosis']
X = df.drop('Diagnosis', axis=1)

#I want to split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)   

#I want to import the KNN pipeline that scales the data itself.
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

#Setting an initial value for k
knn_pipeline.set_params(knn__n_neighbors=5)
knn_pipeline.fit(X_train, y_train)

#Generate the predictions for the test set
y_pred = knn_pipeline.predict(X_test)

#Creating Cross-Validation strategy, giving 5-fold stratified cross-validation.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
#Checking the accuracy of the model using cross-validation
cv_scores = cross_val_score(
    knn_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring='accuracy'
)
print("Cross-Validation scores:", cv_scores)
print("Mean Cross-Validation accuracy:", cv_scores.mean())
print("Cross-Validation standard deviation:", cv_scores.std())

#I am creating a range of k values to test the model's performance with different numbers of neighbors.
k_values = range(1, 31)
#I want to calculate the mean cross-validation accuracy for each k value.
cv_means = []
cv_stds = []

for k in k_values:
    knn_pipeline.set_params(knn__n_neighbors=k)

    scores = cross_val_score(
        knn_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring='accuracy'
    )

    cv_means.append(scores.mean())
    cv_stds.append(scores.std())

#I want to see what the best k value is.
best_k = k_values[cv_means.index(max(cv_means))]
print("Best k:", best_k)
print("Best Cross-Validation accuracy:", max(cv_means))

#I want to plot the k values against their mean cross-validation accuracy
plt.figure(figsize=(10, 6))

plt.errorbar(
    k_values,
    cv_means,
    yerr=cv_stds,
    marker='o',
    capsize=4
)

plt.axvline(
    best_k,
    linestyle='--',
    label=f'Optimal K = {best_k}'
)

plt.xlabel('Number of Neighbours (K)')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.title('KNN Performance Across Different Values of K')

plt.xticks(k_values)
plt.legend()
plt.grid(True)

plt.savefig(
    'plots/knn_k_vs_cross-validation_accuracy.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

#Setting the KNN model to k=9 with the variable best_k = 9.
knn_pipeline.set_params(knn__n_neighbors=best_k)
#Fitting it on all of the training data.
knn_pipeline.fit(X_train, y_train)

#Using test set to evaluate the model's performance.
y_pred = knn_pipeline.predict(X_test)
y_prob = knn_pipeline.predict_proba(X_test)[:, 1]

#Calculating accuracy metrics.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#Add a model performance summary table to plots folder.
performance_results = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, precision, recall, f1]
})

fig, ax = plt.subplots(figsize=(7, 3))

ax.axis('off')

table = ax.table(
    cellText=performance_results.round(4).values,
    colLabels=performance_results.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

plt.savefig(
    'plots/final_model_performance.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()


print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1 Score:", f1)

#I want to generate a classification report for the model's performance on the test set.
print(classification_report(y_test, y_pred))
#I want to save the classification report as a table in a csv file in my data folder.
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('data/classification_report.csv')

#Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
#Display the confusion matrix as a heatmap
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No Diabetes', 'Diabetes']
)

disp.plot()

plt.title('KNN Confusion Matrix')

plt.savefig(
    'plots/knn_confusion_matrix.png',
    dpi=300,
    bbox_inches='tight'
)
plt.close()

#I want to save the k values with their mean cross-validation accruracies.
results = pd.DataFrame({
    'K': list(k_values),
    'Mean CV Accuracy': cv_means
})

top_results = results.sort_values(
    'Mean CV Accuracy',
    ascending=False
)

fig, ax = plt.subplots(figsize=(8, 4))

ax.axis('off')

table = ax.table(
    cellText=top_results.round(4).values,
    colLabels=top_results.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

plt.savefig(
    'plots/knn_k_value_accuracy_results.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()