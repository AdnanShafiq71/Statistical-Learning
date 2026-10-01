import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("age_vs_experience_vs_salary.csv")
data.head()
data_features = ['Age','Experience']
X = data[data_features]
y = data['Salary']

plt.scatter(data['Age'], y)
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs Salary")
plt.savefig("age_vs_salary.png", dpi=150, bbox_inches='tight')
plt.show()

plt.scatter(data['Experience'], y)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
plt.savefig("experience_vs_salary.png", dpi=150, bbox_inches='tight')
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

for feature, coef in zip(data_features, model.coef_):
    print(f"Coefficient for {feature}: {coef}")

y_pred = model.predict(X_test)
print(y_pred)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("R²:", r2)
print("R² (%):", r2 * 100)

prediction = model.predict(pd.DataFrame([[30, 7]], columns=data_features))
print("Predicted salary for age 30, 7 years experience:", prediction[0])


from mpl_toolkits.mplot3d import Axes3D  

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data['Age'], data['Experience'], y, color='blue', label='Actual data')

# Build a grid to draw the regression plane
age_range = np.linspace(data['Age'].min(), data['Age'].max(), 10)
exp_range = np.linspace(data['Experience'].min(), data['Experience'].max(), 10)
age_grid, exp_grid = np.meshgrid(age_range, exp_range)
salary_grid = model.predict(
    pd.DataFrame({'Age': age_grid.ravel(), 'Experience': exp_grid.ravel()})
).reshape(age_grid.shape)

ax.plot_surface(age_grid, exp_grid, salary_grid, alpha=0.4, color='orange')

ax.set_xlabel("Age")
ax.set_ylabel("Experience")
ax.set_zlabel("Salary")
ax.set_title("Age & Experience vs Salary")
plt.savefig("age_experience_vs_salary.png", dpi=150, bbox_inches='tight')
plt.show()
