import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('/content/CustomerData.csv')

df.head()

df.dropna(how='any',inplace=True)
df.drop('CUST_ID',axis=1,inplace=True)