import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

#importing the dataset
dataset=pd.read_csv('Housing_Modified.csv')
X=dataset.iloc[:, 1:12].values
Y=dataset.iloc[:, 0].values
#keeping track of data
column=dataset.columns
column=column.delete([0])

#catagorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
lable=LabelEncoder()
X[:, 4]=lable.fit_transform(X[:, 4])

X[:, 5]=lable.fit_transform(X[:, 5])

X[:, 6]=lable.fit_transform(X[:, 6])

X[:, 7]=lable.fit_transform(X[:, 7])

X[:, 8]=lable.fit_transform(X[:, 8])

ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

X[:, 13]=lable.fit_transform(X[:, 13])
column=column.insert(0,'stories_two')
column=column.insert(0,'stories_three')
column=column.insert(0,'stories_one')
column=column.insert(0,'stories_four')
column=column.delete([7])

#spliting the data into test set and train set
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling ,IN muliple linear regression libraries take care of feature scaling

#model creation
from sklearn.linear_model import LinearRegression
regressior=LinearRegression()
regressior.fit(xtrain, ytrain)

#predcution based on the test
ypred=regressior.predict(xtest)

#applying kfold cross validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=regressior, X=xtrain,y=ytrain,cv=10)
mean_accuracy=accuracies.mean()

#dump model regressior on the disk
pickle.dump(regressior,open('model.pkl','wb'))

#load model from the disk
model=pickle.load(open('model.pkl','rb'))