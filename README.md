# EX 3-Implementation-of-Linear-Regression-Using-Gradient-Descent
## DATE: 07.09.23
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

# Profit Prediction Graph:
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

# Compute Cost Value:
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

# h(x) Value:
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

# Cost Function Using Gradient Descent Graph: 
plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$j(\Theta)$")
plt.title("Cost function using Gradient Descent")

# Profit Prediction Graph:
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

# Profit For The Population 35,000:
def predict(x,theta):

  predictions=np.dot(theta.transpose(),x)

  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

# Profit For The Population 70,000:
predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
### Profit Prediction Graph:
![1](https://github.com/Divya110205/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119404855/5def3de3-c787-4539-b032-753d3f69c915)

### Compute Cost Value:
![2](https://github.com/Divya110205/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119404855/4a8ed7a7-672f-4838-a4f6-b3cd5096941f)

### h(x) Value:
![3](https://github.com/Divya110205/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119404855/cc6e8097-2528-4c9c-870e-a29b24d70f10)

### Cost Function Using Gradient Descent Graph: 
![4](https://github.com/Divya110205/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119404855/6da0e301-c9f2-4264-802f-498cfdcb50dc)

### Profit Prediction Graph:
![5](https://github.com/Divya110205/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119404855/0077dd6c-4d15-4e58-bdcd-da4f94b11da2)

### Profit For The Population 35,000:
![6](https://github.com/Divya110205/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119404855/7f3e10d9-82b7-4ccd-b212-93269247a898)

### Profit For The Population 70,000:
![7](https://github.com/Divya110205/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119404855/5f6a0a22-3353-470e-bd52-ccef72103a4a)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
