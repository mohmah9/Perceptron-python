import numpy as np
import math
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(z, 0)

def gradient_for_relu(z):
    z[z >= 0] = 1
    z[z < 0] = 0
    return z

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

class NeuralNetwork:
    def __init__(self, dimension, hidden_units=100, classes=10):
        self.d = dimension   #784
        self.d_h = hidden_units  #100
        self.k = classes   #10

        np.random.seed(0)
        self.w = np.random.randn(self.d_h, self.d) * math.sqrt(1.0 / self.d)  # 784 weight for every input(hidden) neurons e.g: 100
        self.c = np.random.randn(self.k, self.d_h) * math.sqrt(1.0 / self.d_h)  # 100 weight for every output neurons e.g: 10
        self.accuracy_meter=[]
        self.test_accuracy=0
        self.loss=[]
        self.errors=[]
    def train(self, train_data, learning_rate1 = 0.1, epochs=10):
        X = train_data[0]
        Y = train_data[1]
        descend=(learning_rate1-(learning_rate1/10))/epochs
        for epoch in range(1, epochs + 1):
            print("epochs : ", epoch)
            learning_rate = learning_rate1-((epoch-1)*descend)
            print("LR : ",learning_rate)
            total_correct = 0
            if self.errors !=[]:
                self.loss.append(np.average(self.errors))
            self.errors=[]
            for k in range(X.shape[0]):
                x = X[k]
                y = Y[k]
                
                # Backpropagation.
                sig1, out1, sig2, out2 = self.forward_step(x)
                self.backward_step(learning_rate,x, y, sig1, out1, sig2, out2)

                if np.argmax(out2) == y:
                    total_correct += 1

            acc = total_correct / np.float(X.shape[0])
            self.accuracy_meter.append(acc*100)
            print("epoch {}, training accuracy = {} % \n".format(epoch, acc*100))
        self.loss.append(np.average(self.errors))

    def forward_step(self, x):
        sig1 = np.matmul(self.w, x)
        out1 = relu(sig1)
        # h = sigmoid(z)
        sig2 = np.matmul(self.c, out1)
        out2 = softmax(sig2)
        return sig1, out1, sig2, out2

    def backward_step(self,learning_rate ,x, y, sig1, out1, sig2, out2):
        error = np.zeros(self.k)
        error[y] = 1
        gradient_u = - (error - out2)
        self.errors.append(np.average(np.abs(gradient_u)))
        gradient_u1=gradient_u*gradient_for_relu(sig2)
        gradient_c = np.matmul(gradient_u1[:, np.newaxis], out1[np.newaxis, :])
        delta = np.matmul(self.c.T, gradient_u)
        relu_prime = gradient_for_relu(sig1)
        gradient_b1 = np.multiply(delta, relu_prime)
        gradient_w = np.matmul(gradient_b1[:, np.newaxis], x[np.newaxis, :])

        self.c -= learning_rate * gradient_c
        self.w -= learning_rate * gradient_w
   
    def test(self, test_data):
        X = test_data[0]
        Y = test_data[1]
        total_correct = 0

        for i in range(X.shape[0]):
            x = X[i]
            y = Y[i]
            _, _, _, f = self.forward_step(x)
            if np.argmax(f) == y:
                total_correct += 1
        
        acc = total_correct / np.float(X.shape[0])
        print("testing accuracy = {} %".format(acc*100))
        self.test_accuracy=acc*100
        self.diagram()
    def diagram(self):
        print(self.accuracy_meter,self.test_accuracy)
        t = np.arange(1,len(self.accuracy_meter)+1)
        plt.plot(t,self.accuracy_meter)
        plt.xlabel("epoch")
        plt.ylabel("accuracy %")
        plt.title("test accuracy is : {}".format(self.test_accuracy))
        plt.show()

        print(self.loss)
        plt.plot(np.arange(1,len(self.loss)+1), self.loss)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        # plt.title("loss is : {}".format(self.test_accuracy))
        plt.show()