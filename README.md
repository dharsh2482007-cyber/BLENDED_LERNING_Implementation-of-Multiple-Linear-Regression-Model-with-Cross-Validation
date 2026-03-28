# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries and load the dataset, then preprocess the data by removing unnecessary columns and converting categorical variables into numerical form.



2.Split the dataset into training and testing sets and separate features (X) and target variable (y).



3.Train the Multiple Linear Regression model using the training data and perform cross-validation to evaluate model stability.



4.Predict car prices using the test data and evaluate the model using performance metrics such as MSE, R², and MAE.

## Program:
```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
data= pd.read_csv('CarPrice_Assignment (1).csv')
data.head()

data = data.drop(['car_ID','CarName'], axis=1)
data = pd.get_dummies(data, drop_first=True)
data.head()

X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print('Name: KRITHIKAA P ')
print('Reg. No: 212225040193')
print("\n== Cross-Validation ==")
cv_scores = cross_val_score(model,X,y,cv=5)
print("Fold R^2 scores:", [f"{score:.4f}" for score in cv_scores])
print(f"Average R^2:{cv_scores.mean():.4f}")

y_pred =model.predict(X_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"R^2: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):>10.2f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.grid(True)
plt.show()

## Output:


<img width="852" height="712" alt="image" src="https://github.com/user-attachments/assets/a4a38a94-4b43-46a6-b6c8-03dde09c9e31" />




## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
