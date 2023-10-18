import numpy as np
import matplotlib.pyplot as plt
def write_log(i,cost,theta):
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{i}th itration cost={cost}\n')
def model(X, Y, learning_rate, iteration): 
    m = Y.size 
    theta = np.ones((3, 1))
    for i in range(iteration):
        y_pred = np.dot(x, theta) 
        cost= (1/(2*m))*np.sum(np.square (y_pred - Y))
        d_theta = (1/m) *np.dot (X.T, y_pred - Y)
        theta = theta - learning_rate*d_theta
        write_log(i,cost,theta)
    return theta
data=np.loadtxt(r'C:\Users\bhara\OneDrive\Desktop\python\basic\housingdata.txt',delimiter=',')
log_file_path = 'gradlog.txt'

x=data[:,0:2]
y=data[:,1]
y=y.reshape(y.size,1)
ones_column = np.ones((x.shape[0], 1))
x = np.hstack((ones_column, x))
theta=model(x,y,0.000000001,1000)

a= np.dot([1,1,20000], theta)
print(f'\n pridicted rate for 2000sqft and 1 bedrooms is{a} rupies\n ')
