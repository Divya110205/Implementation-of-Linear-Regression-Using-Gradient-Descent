# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: DIVYA.A
RegisterNumber: 212222230034
*/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('/content/ex1.txt',header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h - y)**2

  return 1/(2*m)*np.sum(square_err)

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    predictions = X.dot(theta)
    error = np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    j_history.append(computeCost(X,y,theta))
  return theta,j_history

theta,j_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$j(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):

  predictions=np.dot(theta.transpose(),x)

  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
### Profit Prediction Graph:
![1](https://github.com/Divya110205/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119404855/07f36042-a753-442a-b95b-9fb0f1b5e28b)

### Compute Cost Value:
![2](https://github.com/Divya110205/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119404855/1f574425-09eb-4f09-8b76-3e140faef3df)

### h(x) Value:
![3](https://github.com/Divya110205/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119404855/3227b8dc-1a1c-4ab3-9229-a1c4cfa5711f)

### Cost Function Using Gradient Descent: 
![4](https://github.com/Divya110205/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119404855/36c50925-5b12-4e8b-a777-d8feb3274d1e)

### Profit Prediction Graph:
![5](https://github.com/Divya110205/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119404855/54370703-0419-4d0d-bce3-939f17cd351a)

### Profit For The Population 35,000:
![6](https://github.com/Divya110205/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119404855/d2770aad-754b-4589-99e5-594b00a3b08c)

### Profit For The Population 70,000:
![7](https://github.com/Divya110205/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119404855/98cdfefa-f6f4-483e-9828-c99b2b99ef1d)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
