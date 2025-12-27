# -*- coding: utf-8 -*-
"""
Created on Wed May 02 17:19:59 2018

@author: User
"""

# 
# data set  Advertising.csv
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# url with dataset
# download the file
data_adv = pd.read_csv('../data/Advertising.csv')
# advertising costs on TV, Radio, and Newspapers 
# column 4 is sales 

X = data_adv.values[:,1:]
y=data_adv.values[:,0] # нулевата колона е пореден номер 
#we have assumed a linear relationship between advertising
#costs on TV and sales
X_TV=X[:,0]                # advertising costs on TV
y=data_adv.values[:,4] # sales 
X_TV = X_TV.reshape(-1, 1)
# y = a X_TV + b 
linreg = LinearRegression()

from sklearn.model_selection import KFold
#kf = KFold(len(X_TV), n_folds=10)
kf = KFold(n_splits=5,shuffle=True)
sm=0
for train_index, test_index in kf.split(X_TV):
    #print('k=',k)
    linreg.fit(X_TV[train_index],y[train_index])
    score_test = linreg.score(X_TV[test_index], y[test_index])
    #print('score_train=',linreg.score(X_TV[train_index], y[train_index])) 
    #print('score_test =',score_test)
    print (score_test)
    if sm < score_test :
        #print('k=',k)
        sm=score_test
        train_minindex = train_index
        test_minindex =  test_index
        
    print
    
# That is the model 
linreg.fit(X_TV[train_minindex],y[train_minindex])
pred_xtest =linreg.predict(X_TV[test_minindex])
print('SCORE on test set: ',linreg.score(X_TV[test_minindex],
                                         y[test_minindex]))
# SCORE on all set 
sctt=linreg.score(X_TV, y)
print('SCORE on all set: ',linreg.score(X_TV,y) )
a=linreg.coef_
b=linreg.intercept_
print('sales = %6.3f X_TV ' %a ,' + %6.3f '  %b)

# Plot test set 
plt.scatter(X_TV[test_minindex], y[test_minindex],  color='black')
plt.plot(X_TV[test_minindex], linreg.predict(X_TV[test_minindex]), color='blue', linewidth=2)
plt.xticks(())
plt.yticks(())
plt.show()

# Plot all set 
plt.scatter(X_TV, y,  color='black') # the all test 
plt.plot(X_TV[test_minindex], linreg.predict(X_TV[test_minindex]), 
         color='blue', linewidth=6)
#plt.plot(X_TV, linreg.predict(X_TV), color='red', linewidth=2)
linreg.fit(X_TV,y)
p = linreg.predict(X_TV)
plt.plot(X_TV, p, color='red', linewidth=2)
plt.xticks(())
plt.yticks(())
plt.show()
