

import pandas as ps


dataset=ps.read_csv("Data.csv")

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

"""Scaling """

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
sc.fit_transform(X)

"""Splitting"""
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3,random_state=21)


"""Decision treee classify"""
from sklearn.tree import DecisionTreeClassifier
dtclassy = DecisionTreeClassifier(criterion="entropy",random_state=1)

dtclassy.fit(xtrain,ytrain)

"""checking its accuracy"""

from sklearn.metrics import accuracy_score,confusion_matrix

ypred=dtclassy.predict(xtest)

print(confusion_matrix(ypred,ytest))
print(accuracy_score(ypred,ytest))



