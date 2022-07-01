

import numpy as np


X = np.genfromtxt('csv_dataset_MNIST/train_images.csv', delimiter=',', dtype=None)
y = np.genfromtxt('csv_dataset_MNIST/train_labels.csv', delimiter=',', dtype=None)



nb_classes = 10
targets = y.reshape(-1)
y = np.eye(nb_classes)[targets]

class layer:
    def __init__(self, weights, neurons):
        self.neurons = np.random.randn(weights, neurons)
        self.bias = np.zeros((1,neurons))
    def forward(self, X):
        self.Z = np.dot(X, self.neurons) + self.bias

class activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        normalization = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.A = normalization

class cost:
    def cost_mse(self, y_hat, y):
        Error = (1/60000)*np.sum(((y - y_hat)**2))
        print(Error)

class backprop:
    def forward(self, learning_rate):
        softmaxA.forward(a3)
        functionA3 = softmaxA.A
        theta3 = layer3.neurons
        delta3 = (1/60000) * np.dot((y - a3), functionA3.T)
        error3 = np.sum(np.dot(delta3.T, a2))
        theta_grad3 = theta3 + (-learning_rate*error3)
        layer3.neurons = theta_grad3

        softmaxA.forward(a2)
        functionA2 = softmaxA.A
        theta2 = layer2.neurons
        theta2f = np.dot(theta2, functionA2.T)
        delta2 = np.dot(delta3, theta2f.T) 
        error2 = np.sum(np.dot(delta2.T, a1))
        theta_grad2 = theta2 + (-learning_rate*error2)
        layer2.neurons = theta_grad2

        softmaxA.forward(a1)
        functionA1 = softmaxA.A
        theta1 = layer1.neurons
        theta1f = np.dot(theta1, functionA1.T)
        delta1 = np.dot(delta2, theta1f.T) 
        error1 = np.sum(np.dot(delta1.T, X))
        theta_grad1 = theta1 + (-learning_rate*error1)
        layer1.neurons = theta_grad1



layer1 = layer(784, 500)
layer2 = layer(500, 250)
layer3 = layer(250, 10)
softmaxA = activation_Softmax()


i = 0
while i < 20:
    layer1.forward(X)
    softmaxA.forward(layer1.Z)
    a1= softmaxA.A
    layer2.forward(softmaxA.A)
    softmaxA.forward(layer2.Z)
    a2= softmaxA.A
    layer3.forward(softmaxA.A)
    softmaxA.forward(layer3.Z)
    a3 = softmaxA.A
    
    backpropagation = backprop()
    backpropagation.forward(0.1)
    
    mse = cost()
    mse.cost_mse(a3, y)
    i += 1
