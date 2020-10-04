import numpy as np
from matplotlib import pyplot as plt

X=np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])
y = np.array([0,0,1,0])

def activation_func(n):
    if n>=0:
        return 1
    else:
        return 0

def perceptron_gd_plot(X, Y):

    w = np.array([0.1,0.2,0.3])
    eta = 0.1
    epochs = 15
    errors = []
    
    xin = X.copy()
    Y [Y==0]=-1
    for _ in range(epochs):
        yin = np.dot(xin,w)
        e = Y-yin
        errors.append(np.abs(np.average(e)))
        grad = eta *e
        t=np.rot90(xin)
        t= np.dot(t,grad)
        t=np.flip(t)
        w = w + t
    plt.xlabel('X1')
    plt.ylabel('X2')
    for d, sample in enumerate(X):
        plt.scatter(sample[0], sample[1],color="black", s=120, marker='.')
    print("activation function response : ")
    for i, x in enumerate(X):
        print(x," : ",activation_func(x[0]*w[0] + x[1]*w[1] + w[2]))
    x=np.arange(0,1.5,0.01)
    y= -(w[0]/w[1])*x - (w[2]/w[1])
    plt.plot(x,y)
    plt.xlabel("X1")
    plt.ylabel('X2')
    plt.show()

    plt.plot(np.arange(len(errors)),errors)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.show()

perceptron_gd_plot(X,y)