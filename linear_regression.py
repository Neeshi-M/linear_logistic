import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df =pd.read_csv('1.01. Simple linear regression.csv')

x=[]
y=[]
with open('/Users/jaineel/Downloads/1.01. Simple linear regression.csv') as csvdata:
    items=csv.reader(csvdata)
    for row in items:
        x.append(row[0])
        y.append(row[1])
#print(x)
#print(y)
final_y=[]
final_x=[]

for i in range(1,len(x)):
    final_y.append(x[i])



for j in range(1,len(x)):
    final_x.append(y[j])






print(final_y)

print(final_x)
x=np.array(final_x)
y=np.array(final_y)
y=y.astype('float')/1000
y=y.reshape(y.size,1)

#plt.scatter(x,y )
#plt.show()


x=np.vstack((np.ones((x.size,)),x)).T
print(x.shape)
print(y.shape)
def model(x,y,learningrate,iteration):
   m=y.size
   theta=np.zeros((2,1))
   for i in range(iteration):
       y_pred=np.dot(x,theta)
       cost=(1/(2*m))*np.sum(np.square(y_pred-y))
       d_theta=(1/m)*np.dot(x.T,y_pred-y)
       theta=theta-learningrate*d_theta

   return theta
iterations=10000
learning_rate=0.08
final_theta=model(x.astype('float'),y.astype('float'),learning_rate,iterations)
final_theta=final_theta.astype('float')*1000


print(np.dot([1,3.17],final_theta))
