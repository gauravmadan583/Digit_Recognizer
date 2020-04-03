import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Data loading
data = open("train.csv")

f=data.readlines()

count = 0

for i in f:
    count = count +1

images= np.zeros((count-1,784), dtype = np.uint8)
label= np.zeros((count-1,1), dtype = np.uint8)


for i in range(count-1):
    SS = f[i+1].split(",")
    label[i] = SS[0]
    images[i] = SS[1:]

# Image formation with help of data
for i in range(10):
    print(label[i])
    img = np.reshape(images[i],(28,28))
    plt.imshow(img,interpolation='nearest')
    plt.show()

images = np.concatenate((np.ones((np.shape(images)[0],1)), images), axis =1 )

# Cost Function returning Cost and gradient
def costFunction(X,y,theta,num_labels,lmb):
    m = np.shape(X)[0]
    J = 0
    grad = np.zeros(np.shape(theta))
    for i in range(num_labels):
        J = J - ( y[:,i].transpose()@np.log(sigmoid(X@theta[:,i])) + (1-y[:,i]).transpose()@np.log(1-sigmoid(X@theta[:,i])) ) + lmb*(theta[:,i].transpose()@theta[:,i])/2

    grad = X.transpose()@(sigmoid(X@theta)-y)/m + lmb*theta
    grad[0] = grad[0] - lmb*theta[0]

    return J/m,grad

#Sigmoid function
def sigmoid(X):
    X=np.exp(X)
    X=X/(X+1)
    return X

#Gradient Descent
def gradientDescent(X,y,theta,num_labels,iterations,alpha,lmd):
    J_history = np.zeros(iterations)

    for i in range(iterations):
        J_history[i],grad=costFunction(X,y,theta,num_labels,lmd)
        theta = theta - alpha*grad
        print(i," ",J_history[i])

    return theta,J_history

#Precision
def precision(X,y):
    tp=0
    fp=0
    for i in range(np.shape(X)[0]):
         j,k = np.unravel_index(X[i].argmax(), np.shape(X))
         # print(j)
         if k==label[i]:
             tp = tp + 1
         else:
            fp = fp + 1

    return tp*100/(tp+fp)


#Prediction
def predict(X):
    y = np.zeros((np.shape(X)[0],1))
    for i in range(np.shape(X)[0]):
        j,y[i] = np.unravel_index(X[i].argmax(), np.shape(X))

    return y


#main function
num_labels = 10
theta = np.zeros((np.shape(images)[1],num_labels))
y= np.zeros((np.shape(images)[0],num_labels))

a = [0,1,2,3,4,5,6,7,8,9]

for i in range(count-1):
    y[i] = (a==label[i])

print(costFunction(images,y,theta,num_labels))

final_theta,J_history=gradientDescent(images,y,theta,num_labels,400,0.00001,100)

print(precision(sigmoid(images@final_theta),y))

#plotting cost history
plt.plot(J_history)
plt.show()

#choosing regularisation constant
lmds = [ 0, 0.1, 0.3, 1, 3, 10, 30, 100]
p = np.zeros(np.shape(lmds))
final_J = np.zeros(np.shape(lmds))

for i in range(len(lmds)):
    print("lambda = ", lmds[i])
    final_theta,J_history = gradientDescent(images,y,theta,num_labels,400,0.00001,lmds[i])
    p[i]= precision(sigmoid(images@final_theta),y)
    final_J = J_history[-1]

plt.plot(lmds,p)
plt.show()

#predicting on test set
test = open("test.csv")

lines = test.readlines()
count_test = 0
for i in lines:
    count_test = count_test + 1
print(count_test)
images_test = np.zeros((count_test-1,784), dtype=np.uint8)
for i in range(count_test-1):
    SS = lines[i+1].split(",")
    images_test[i] =  SS[0:]
print(np.shape(images_test))

for i in range(10):
    img = np.reshape(images_test[i],(28,28))
    plt.imshow(img,interpolation='nearest')
    plt.show()
print(np.shape(images_test))
images_test = np.concatenate((np.ones((np.shape(images_test)[0],1)),images_test),axis=1)
print(np.shape(images_test))

y_test = predict(sigmoid(images_test@final_theta))

import csv

with open('submission.csv', mode='w', newline='') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='y', quoting=csv.QUOTE_MINIMAL)
    employee_writer.writerow(['ImageId', 'Label'])
    for i in range(count_test-1):
        employee_writer.writerow([i+1, y_test[i][0]])
