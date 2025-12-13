# -*- coding: utf-8 -*-
"""
Created on Wed May 02 17:18:17 2018

@author: User
"""

# 
# data  Advertising.csv
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# url with dataset
# download the file
data_adv = pd.read_csv('Advertising.csv')
# advertising costs on TV, Radio, and Newspapers 
# The sales are in column 4  

X = data_adv.values[:,1:]

#we have assumed a linear relationship between advertising
#costs on TV and sales
X_TV=X[:,0]     # 0 = advertising costs on TV
                           # 1 = advertising costs on Radio
y=data_adv.values[:,4]    # sales 

# y = a X_TV + b 
linreg = LinearRegression()
# Train the model using the all set 
X_TV = X_TV.reshape(-1, 1)
linreg.fit(X_TV,y)
p = linreg.predict(X_TV)
sc=linreg.score(X_TV, y)
print('score=',sc)

a=linreg.coef_
b=linreg.intercept_
print('sales = %6.3f X_TV ' %a ,' + %6.3f '  %b)
#print('sales = %6.3f X_TV ' %a[0],' + %6.3f Radio '  %a[1]  ,' + %6.3f '  %b)


# Plot outputs

plt.scatter(X_TV, y,  color='black')
plt.plot(X_TV, p, color='blue', linewidth=2)
plt.xticks(())
plt.yticks(())
plt.show()




#------------
print('Train-Test Subsets')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_TV, y, test_size=0.3)
linregTT = LinearRegression()
# Train the model using the train subset 

linregTT.fit(X_train,y_train)
p = linregTT.predict(X_test)
sc=linregTT.score(X_test, y_test)
print('score=',sc)
a=linregTT.coef_
b=linregTT.intercept_
print('sales = %6.3f X_train ' %a ,' + %6.3f '  %b)

# Plot test set 
y_pred=linregTT.predict(X_test)
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=2)
plt.xticks(())
plt.yticks(())
plt.show()






