#####    multiple Linear Regression 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('/content/50_Startups.csv')


df.head()


#dummies works on only categorical data
df_dummies=pd.get_dummies(df,drop_first=True)

#Dividing data
    
x_mlr=df_dummies.drop(['Profit'],axis=1)
y_mlr=df_dummies.Profit


## Train and Test data
from sklearn.model_selection import train_test_split
X_tesr_mlr,X_test_mlr,y_train_mlr,y_test_mlr=train_test_split(x_mlr,y_mlr,test_size=0.2)


## Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_mlr=sc.fit_transform(X_train_mlr)
X_test_mlr =sc.transform(X_test_mlr)


## Create the MLR Model
from sklearn.linear_model import LinearRegression
reg2=LinearRegression()
reg2.fit(X_train_mlr,y_train_mlr)


y_pred2=reg2.predict(X_test_mlr)



### Mean Absolute error with multile features
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test_mlr,y_pred2)   #error value has increased
 

 ####using correlation we can find relevant features
corr_matrix=df_dummies.corr()['Profit'] 