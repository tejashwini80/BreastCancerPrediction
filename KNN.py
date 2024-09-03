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

#Model building for KNN
from sklearn.neighbors import KNeighborsClassifier

#1.Initiating the classifier
model_knn=KNeighborsClassifier()
#2.Passing the data tho classifier
model_knn.fit(x_train_sc,y_train)

#Predicting
y_pred_knn=model_knn.predict(x_test_sc)

print(y_pred_knn)


#Classification metrics to see how the model behaves
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred_knn)*100)


#predict for a new patient
new_data=[[19.99,20.38,122.8,2001,0.1184,
           0.2776,0.4001,0.1471,0.2419,0.07871,4.095,
           0.5053,8.589,153.4,0.002399,0.04904,0.08373,
           0.07587,0.03003,0.006193,25.38,13.33,984.6,2919,
           0.1622,0.6656,0.7119,0.2954,0.8601,0.1089]]

#Feature scaling for the new data
new_data_sc=sc.transform(new_data)

y_pred_new=model_knn.predict(new_data_sc)

print(y_pred_new)                                       #95








