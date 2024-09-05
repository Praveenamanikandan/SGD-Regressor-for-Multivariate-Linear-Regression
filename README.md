# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start Step
2.Data Preparation
3.Hypothesis Definition
4.Cost Function
5.Parameter Update Rule
6.Iterative Training
7.Model Evaluation
8.End

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Praveena M
RegisterNumber:  212223040153
*/
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


```
```
data = fetch_california_housing()
```
```
print(data)
```
## Output:
![image](https://github.com/user-attachments/assets/3b1e6640-7099-43c5-b989-aa103acb6556)
```
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
```
## Output:
![image](https://github.com/user-attachments/assets/c07bbf72-b9ea-4ade-b50b-814fe9a1ebc6)
```
df.info()

```
## Output:
![image](https://github.com/user-attachments/assets/92b2f957-f183-42eb-b398-b85d3ac52dee)
```
X=df.drop(columns=['AveOccup','target'])

```
```
X.info
```
## Output:
![image](https://github.com/user-attachments/assets/3287ab38-ce47-4e96-9a42-898dd060157a)
```
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
```
```
X.head()
```
## Output:

![image](https://github.com/user-attachments/assets/d6bacdfa-b268-4487-aea2-b973c7dcb6a1)
```
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.fit_transform(Y_test)
```
```
print(X_train)
```
## Output:
![image](https://github.com/user-attachments/assets/685c1847-b2be-49f0-8f54-524c7979e158)

```
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
```
```
multi_output_sgd = MultiOutputRegressor(sgd)

```
```
multi_output_sgd.fit(X_train, Y_train)
```
## Output:

![image](https://github.com/user-attachments/assets/f6f2d933-5f53-4398-9d57-521e28953928)
```
Y_pred = multi_output_sgd.predict(X_test)
 ```
```
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
```
```
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
```
## Output:

![image](https://github.com/user-attachments/assets/e80f55bf-8f61-41ab-83d4-002f5ae8b28b)

```
print("\nPredictions:\n", Y_pred[:5])
```
## Output:

![image](https://github.com/user-attachments/assets/d3e59262-425d-4772-904a-400d8d7d8ce4)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
