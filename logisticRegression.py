

import pandas as ps

"""importing data set"""
dataset=ps.read_csv("Data.csv")

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values



"""Splitting dataset training set"""
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.5,random_state=21)

"""Scaling value due to non relational magnitude"""
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)



"""Scaling before splittng and after splitting also same No change in accuracy score"""

#print(xtrain)

"""creating logistic regression obj and invoking"""
from sklearn.linear_model import LogisticRegression

logiRegression=LogisticRegression(random_state=21)

logiRegression.fit(xtrain,ytrain)

"""Checking accuracy"""
from sklearn.metrics import accuracy_score,confusion_matrix

ytestPredict=logiRegression.predict(xtest)

print("confusion_matrix :",confusion_matrix(ytest,ytestPredict))

print("accuracy_score :",accuracy_score(ytest,ytestPredict))

