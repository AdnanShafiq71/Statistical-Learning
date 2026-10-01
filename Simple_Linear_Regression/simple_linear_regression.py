import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("experience_vs_salary.csv")

X = data[['YearsExperience']]
y = data['Salary']

plt.scatter(X, y)

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Years of Experience vs Salary")

plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

y_pred = model.predict(X_test)
print(y_pred)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("R²:", r2)
print("R² (%):", r2 * 100)

prediction = model.predict([[7]])

print("Predicted salary for 7 years of experience:", prediction[0])

plt.scatter(X_test, y_test, label="Actual")

plt.plot(X_test, y_pred, label="Regression line")

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression")

plt.legend()
plt.show()