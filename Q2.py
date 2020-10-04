import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

a = pd.read_csv("01_heights_weights_genders.csv", sep=',',header=None).values
a=a[1:]
a=np.where(a=="Male", 1, a)
a=np.where(a=="Female", -1, a)
# print(a)
# np.random.shuffle(a)
a=np.insert(a,3,1,axis=1)
b=np.rot90(a)
c=b[:-1]
c=np.rot90(c)
c=np.rot90(c)
c=np.rot90(c)
X=c
Y=b[3]
X=X.astype(float)
Y=Y.astype(float)

def activation_func(n):
    f = np.tanh(n)
    return f
def perceptron_gd_plot(X, Y):
    w = np.array([0.2, 0.2, 0.2])
    eta = 0.1
    epochs = 120

    errors = []

    xin = X.copy()
    sig=xin[1].max()
    xin=np.rot90(xin)
    xin[1]=xin[1]/xin[1].max()
    xin[2]=xin[2]/xin[2].max()
    xin=np.rot90(xin)
    xin=np.rot90(xin)
    xin=np.rot90(xin)
    for z in range(epochs):
        yin = np.dot(xin, w)
        e = Y - activation_func(yin)
        errors.append(np.abs(np.average(e)))
        grad = eta * e
        t = np.rot90(xin)
        t = np.dot(t, grad)
        t = np.flip(t)
        w = w + t

    return w,sig,errors

weight,sig,errors=perceptron_gd_plot(X, Y)
loss_unit_number=0

for i in range(len(X)):
    y= -((weight[0] / weight[1]) * X[i][0] - (weight[2] / weight[1]))+sig
    if X[i][1] > y and Y[i] == -1:
        loss_unit_number+=1
    if X[i][1] < y and Y[i] == 1:
        loss_unit_number += 1
print(loss_unit_number)
fig = plt.figure()
ax = fig.add_subplot(1, 1,1)
print("please wait for plotting ...")
for i in range(len(X[:2000])):
    if i>1000:
        i+=4000
    if Y[i] == 1:
        ax.scatter(X[i][0], X[i][1], alpha=0.8,color="blue", edgecolors='none', s=30)
    if Y[i] == -1:
        ax.scatter(X[i][0], X[i][1], alpha=0.8, color="pink", edgecolors='none', s=30)
x = np.arange(50, 80, 0.01)
y = -((weight[0] / weight[1]) * x - (weight[2] / weight[1]))+sig

plt.plot(x,y)
plt.title('Matplot scatter plot')
plt.xlabel("Weight")
plt.ylabel("Height")
plt.show()

plt.plot(np.arange(len(errors)),errors)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss")
plt.show()
