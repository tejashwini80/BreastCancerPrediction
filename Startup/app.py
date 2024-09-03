import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_csv('/content/50_Startups.csv')


df.head()

#dividing the data as x and y 
#SLR   -1 col
x=df.iloc[:,0].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2)

#reshaping the x variables to 2d array
x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)

#Feature scaling '''here it is not so imp bcz it has only 1 feature
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

##Creating LR model
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)


#Predicting
y_pred=reg.predict(x_test)


#Plotting actual vs predicting
plt.plot(y_test,color='blue',label='test')
plt.plot(y_pred,color='red',label='predictions')
plt.show()
