# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 13:43:02 2023

@author: Ivan
"""

# ExampleEqNum_framinghamLR
import pandas as pd
# Load the dataset
dM = pd.read_csv("framingham.csv")


# намиране на редове с NAN стойности 
dM.isnull().sum()
# премахвам  липсващите стойности в същия файл  
dM.dropna(inplace=True)
##
# 
# равен брой наблюдения само  в тренировъчното  
import pandas as pd
dM['TenYearCHD'].value_counts()
#  0    3101
#  1     557
#
print (dM.head())
dM1=dM[dM['TenYearCHD'] == 0 ]  
dM2=dM[dM['TenYearCHD'] == 1 ]
#
dMBigger = dM2
dMSmall=dM1
if len(dM1) > len(dM2):
    dMBigger = dM1
    dMSmall=dM2
    
# разделяме малкото множество на тренировъчно и тестово подмножество
from sklearn.model_selection import train_test_split
dMSmall_train, dMSmall_test = train_test_split(dMSmall, test_size = 0.40) 
#  100 - len(dMBigger)
#  x   --  len(dMSmall_train)
#  len(dMSmall_train) < len(dMBigger)
#
# процента на броя на елементите от тренировъчното подможество dMSmall_train
#от големината голямото подмножество 
xpro_dMBigger_tr=float(len(dMSmall_train))/float(len(dMBigger))
print (xpro_dMBigger_tr)
# разделяме голямото подмножество с този процент 
# 
dMBigger_train,dMBigger_te =train_test_split(dMBigger, test_size=1-xpro_dMBigger_tr,
                                             random_state=56)
#обединяваме двет етренировъчни подмножество, който имат по равен брой елементи
#Получаваме общото тренировъчно подмножество
dMG_tr=pd.concat([dMSmall_train, dMBigger_train])
dMG_test = pd.concat([dMSmall_test,dMBigger_te])
print('values \n', dMG_tr['TenYearCHD'].value_counts() ) 
#
# files dMG_tr, dMG_test
X_train = dMG_tr.values[:,0:8]
y_train = dMG_tr.values[:,8]
#
X_test = dMG_test.values[:,0:8]
y_test = dMG_test.values[:,8]
#
from sklearn.linear_model import LogisticRegression
classLogR = LogisticRegression(solver='lbfgs', max_iter=5000, class_weight='balanced')

classLogR =LogisticRegression(solver='liblinear',  class_weight="balanced", random_state=56) 

classLogR.fit(X_train, y_train)

y_pred = classLogR.predict(X_test)
print ('NEW APPROACH...')
print ('TRAIN set score :',classLogR.score(X_train, y_train))
print ('TEST set score  :',classLogR.score(X_test, y_test))

#
#Confusiton matrix and clasification report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print ('\n confussion matrix for the test set :\n',confusion_matrix(y_test,
                                                                    y_pred))
print ('\n classification report for the test set :\n',
       classification_report(y_test, y_pred))




