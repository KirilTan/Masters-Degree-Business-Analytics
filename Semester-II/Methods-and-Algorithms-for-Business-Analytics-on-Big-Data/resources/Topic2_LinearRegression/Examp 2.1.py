# -*- coding: utf-8 -*-
"""
Created on Wed May 02 17:16:04 2018

@author: User
"""

# 
import numpy as np 
import matplotlib.pyplot as plt
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])


# Plot outputs
plt.scatter(x, y,  color='black')
plt.show()


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
x = x.reshape(-1, 1)
linreg.fit(x,y)
y_pred = linreg.predict(x)
sc=linreg.score(x, y)
print('score=',sc) 
a=linreg.coef_
b=linreg.intercept_
print('y = %8.5f x ' %a ,' + %6.3f '  %b)

# Plot outputs
plt.scatter(x, y,  color='black')
plt.plot(x, y_pred, color='blue', linewidth=2)
plt.show()
