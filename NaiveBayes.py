import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('./data.csv')

df.head(5)
#dignosing the B and M values
df.diagnosis.value_counts()/len(df) *100

#splitting the data into x and y
#x is everything except first 2 colunms
x=df.drop(['id','diagnosis'],axis=1)
y=df.diagnosis.values

# Training and Testing split
import sklearn
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.25)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_sc=sc.fit_transform(x_train)
x_test_sc=sc.transform(x_test)


##### NaiveBayes-----------------------------------------------------------------
from sklearn.naive_bayes import GaussianNB
#1.initiating
model_nb=GaussianNB()
#2.passing the data to it
model_nb.fit(x_train_sc,y_train)

#Predicting
y_pred_nb=model_nb.predict(x_test_sc)

#Classification metrics to see how the model behaves
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_nb)*100)                      #90