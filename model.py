# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:52:54 2020

@author: Mukund Rastogi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('AB_NYC_2019.csv',encoding= 'unicode_escape')
#data.head()

data.isnull().sum()

data['reviews_per_month'].fillna(data['reviews_per_month'].mean(), inplace=True)

Airbnb_data=data.drop(columns=['id','name','host_id','host_name','last_review','latitude','longitude','reviews_per_month','neighbourhood'])
Airbnb_data=Airbnb_data.dropna()

#Airbnb_data.head()
#Airbnb_data.columns
'''
for i in range(0,len(Airbnb_data)):
    if(Airbnb_data['room_type'][i]=='Private room'):
        Airbnb_data['room_type'][i]=1
    elif Airbnb_data['room_type'][i]=='Entire home/apt':
        Airbnb_data['room_type'][i]=2
    else:
        Airbnb_data['room_type'][i]=3
        
for i in range(0,len(Airbnb_data)):
    if Airbnb_data['neighbourhood_group'][i]=='Brooklyn':
       Airbnb_data['neighbourhood_group'][i]=1
    elif Airbnb_data['neighbourhood_group'][i]=='Manhattan':
        Airbnb_data['neighbourhood_group'][i]=2 
    elif Airbnb_data['neighbourhood_group'][i]=='Bronx':
        Airbnb_data['neighbourhood_group'][i]=3
    elif Airbnb_data['neighbourhood_group'][i]=='Queens':
        Airbnb_data['neighbourhood_group'][i]=4
    elif Airbnb_data['neighbourhood_group'][i]=='Staten Island':
        Airbnb_data['neighbourhood_group'][i]=5
        
'''

Airbnb_data.loc[Airbnb_data['room_type']=='Private room','room_type']=1
Airbnb_data.loc[Airbnb_data['room_type']=='Entire home/apt','room_type']=2
Airbnb_data.loc[Airbnb_data['room_type']=='Shared room','room_type']=3

Airbnb_data.loc[Airbnb_data['neighbourhood_group']=='Brooklyn','neighbourhood_group']=1
Airbnb_data.loc[Airbnb_data['neighbourhood_group']=='Manhattan','neighbourhood_group']=2
Airbnb_data.loc[Airbnb_data['neighbourhood_group']=='Bronx','neighbourhood_group']=3
Airbnb_data.loc[Airbnb_data['neighbourhood_group']=='Queens','neighbourhood_group']=4
Airbnb_data.loc[Airbnb_data['neighbourhood_group']=='Staten Island','neighbourhood_group']=5

#Airbnb_data.head()
#Airbnb_data=Airbnb_data.apply(pd.to_numeric)

import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
corr = Airbnb_data.corr()
sns.heatmap(corr, annot=True, annot_kws={"size": 9}, cmap = sns.color_palette("PuOr_r", 50), 
                     vmin = -1, vmax = 1)

from sklearn.model_selection import train_test_split
y=Airbnb_data['price']
x=Airbnb_data.drop('price',axis=1)
X = x.apply(pd.to_numeric, errors='coerce')
Y = y.apply(pd.to_numeric, errors='coerce')
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.3, random_state = 42)

from sklearn.linear_model import LinearRegression as lm
from math import sqrt
regressor=lm().fit(xTrain,yTrain)
predictions=regressor.predict(xTest)

from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error: %.2f"% mean_squared_error(yTest,predictions))
print("R-square: %.2f" % r2_score(yTest,predictions))

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(xTrain,yTrain)
predictions_ridge=ridge_regressor.predict(xTest)

print("Mean squared error: %.2f"% mean_squared_error(predictions_ridge,yTest))
print("R-square: %.2f" % r2_score(yTest,predictions_ridge))

#Using Lasso
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(xTrain,yTrain)
predictions_lasso=ridge_regressor.predict(xTest)

print("Mean squared error: %.2f"% mean_squared_error(predictions_lasso,yTest))
print("R-square: %.2f" % r2_score(yTest,predictions_lasso))

from sklearn.tree import DecisionTreeRegressor
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(xTrain,yTrain)
y_Dtree=DTree.predict(xTest)
score_Dtree=DTree.score(xTest, yTest)
print('R-squared score (training): {:.3f}'.format(DTree.score(xTrain, yTrain)))
print('R-squared score (test): {:.3f}'.format(DTree.score(xTest, yTest)))


'''
import pickle
# Saving model to disk
pickle.dump(lasso_regressor, open('model.pkl','wb'))
#print(xTest.info())
#print(yTest.info())
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
model.score(xTest,yTest)
print(model.predict([[2,3,6,4,5,50]]))
'''
