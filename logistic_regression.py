import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


purchase_data =pd.read_csv("User_Data - User_Data.csv")
#print(purchase_data.head(10))
#sns.count_
gender=pd.get_dummies(purchase_data["Gender"],drop_first=True)
#purchase=pd.get_dummies(purchase_data["Purchased"],drop_first=True)
purchase_data=pd.concat([purchase_data,gender],axis=1)
#print(purchase_data.head(10))
purchase_data.drop(['Gender'],axis=1,inplace=True)
#print(purchase_data.head(10))
y=purchase_data['Purchased']
x=purchase_data.drop('Purchased',axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
#print(x_train)
from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
predictions=logmodel.predict(x_test)
from sklearn.metrics import classification_report
classification_report(y_test,predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
from  sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
