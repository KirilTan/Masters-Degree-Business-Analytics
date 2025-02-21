# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 19:05:50 2018

@author: User
"""



# въвеждане на данни   
import numpy as np 

u = [0, 1.2, 2.4, 3.2, 4, 5, 6, 7, 8, 9]
v = [1.5, 2.8, 2, 5, 7, 8, 8.5, 9.2, 10.4, 12.5]

print u,v 
for i in range(0,10):
    print (u[i],' ',v[i])



#---------------
x = np.array([0, 1.2, 2.4, 3.2, 4, 5, 6, 7, 8, 9])
y = np.array([1.5, 2.8, 2, 5, 7, 8, 8.5, 9.2, 10.4, 12.5])

print (x)

print (y)

for i in range(0,10):
    print (' i=',i,' ',  x[i],' ',y[i]   )


#---------------
#-------------
#-----------
import pandas
url = "insurance.csv"
dataset = pandas.read_csv(url)
# dataset = pandas.read_csv("insurance.csv")
print(dataset)
data = dataset.values
x=data[:,0]
y=data[:,1]
for i in range(0,63):
    print (x[i],' ',y[i])

#------------

print
print (' PANDAS - Load *.txt')
import pandas
#datasetTXT = pandas.read_csv("AutoInsurSweden.txt", delim_whitespace=True, decimal=',')
datasetTXT = pandas.read_csv('AutoInsurSweden.txt', delim_whitespace=True, decimal=',')
data = datasetTXT.values
x=data[:,0]
y=data[:,1]
for i in range(0,63):
    print (x[i],' ',y[i])


#------------

print
print (' NUMPY - Load *.txt')
import numpy 
data = numpy.loadtxt('ex2data1.txt', delimiter = ',')
x = data[:,0:2]
y = data[:,2]
for i in range(0,100):
    print (x[i],' ',y[i])





