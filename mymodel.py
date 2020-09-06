# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


da=pd.read_csv('student.csv')
da.head()


da['sub1'].fillna(da['sub1'].mean(), inplace=True)
da['sub2'].fillna(da['sub2'].mean(), inplace=True)
da['sub3'].fillna(da['sub3'].mean(), inplace=True)
da['study_hours'].fillna(da['study_hours'].mean(), inplace=True)
da['average']=(da['sub1']+da['sub2']+da['sub3'])/3

da.head()

da['study_hours']=da['study_hours'].astype('int64')
da['marks']=da['marks']-10

da.head()
da['assignments']=pd.get_dummies(da['assignments'])

da['marks']=da['marks']+(da['assignments']*10)
da.head()

da['assignments']=da['assignments'].astype('int64')

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(da.drop(['sub1','sub2','sub3','marks'],axis='columns'),da.marks)

pickle.dump(regressor, open('mymodel.pkl','wb'))

mymodel = pickle.load(open('mymodel.pkl','rb'))
print(mymodel.predict([[10,1,8]]))











