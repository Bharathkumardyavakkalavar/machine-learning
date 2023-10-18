import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=np.loadtxt(r'C:\Users\bhara\OneDrive\Desktop\all about ml\place.txt',delimiter=',')
""" print(data.shape)
plt.scatter(data[:,0],data[:,1])
plt.xlabel("cgpa")
plt.ylabel("package")
plt.show() """
cgpa=data[:,0].reshape(-1, 1)
package=data[:,1]
lr=LinearRegression()
ctrain,ctest,ptrain,ptest=train_test_split(cgpa,package,test_size=0.2,random_state=2)
lr.fit(ctrain,ptrain)
a=lr.predict(ctest[0].reshape(1,1))

""" plt.scatter(data[:,0],data[:,1])
plt.plot(ctrain,lr.predict(ctrain),color='r')
plt.xlabel("cgpa")
plt.ylabel("package")
plt.show()
"""
m=lr.coef_
c=lr.intercept_
print(m,c)

