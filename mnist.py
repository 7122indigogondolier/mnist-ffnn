"""
Author: Utkrist P. Thapa '21
CSCI 315, Artificial Intelligence
Project 2: Implementing a feed forward neural network using the MNIST dataset and
OOP in Python
This network can be trained to recognize handwritten digits from the MNIST dataset
"""

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import math
import matplotlib.pyplot as plt
import random
from keras.datasets import mnist 

class NeuralNetwork:
    def __init__(self, num_inputs, num_hiddenNodes1, num_hiddenNodes2, num_outputs):
        # attributes of the object
        self.num_inputs = num_inputs
        self.num_hiddenNodes1 = num_hiddenNodes1
        self.num_hiddenNodes2 = num_hiddenNodes2
        self.num_outputs = num_outputs
        self.lr = 0.1

        # define weight matrices
        self.weights_ih1 = np.zeros((self.num_hiddenNodes1, self.num_inputs))
        self.weights_h1h2 = np.zeros((self.num_hiddenNodes2, self.num_hiddenNodes1))
        self.weights_h2o = np.zeros((self.num_outputs, self.num_hiddenNodes2))

        # define bias matrices
        self.bias_ih1 = np.zeros((self.num_hiddenNodes1, 1))
        self.bias_h1h2 = np.zeros((self.num_hiddenNodes2, 1))
        self.bias_h2o = np.zeros((self.num_outputs, 1))
        

    def setup(self): 
        # initialize the weight and bias matrices for:
        #       1) input --> hidden1
        for i in range(self.num_hiddenNodes1):
            for j in range(self.num_inputs):
                self.weights_ih1[i][j] = np.random.normal(0, self.num_inputs ** (-0.5))
            self.bias_ih1[i][0] = np.random.normal(0, self.num_inputs ** (-0.5))

        #       2) hidden1 --> hidden2
        for i in range(self.num_hiddenNodes2):
            for j in range(self.num_hiddenNodes1):
                self.weights_h1h2[i][j] = np.random.normal(0, self.num_hiddenNodes1 ** (-0.5))
            self.bias_h1h2[i][0] = np.random.normal(0, self.num_hiddenNodes1 ** (-0.5))

        #       3) hidden2 --> output
        for i in range(self.num_outputs):
            for j in range(self.num_hiddenNodes2):
                self.weights_h2o[i][j] = np.random.normal(0, self.num_hiddenNodes2 ** (-0.5))
            self.bias_h2o[i][0] = np.random.normal(0, self.num_hiddenNodes2 ** (-0.5))
                
    def feedforward(self, inputs):
        # feeding forward to:
        #       1) input --> hidden1
        inputs = np.asarray(inputs)     # preparing the input into the appropriate matrix form
        inputs = np.reshape(inputs, (self.num_inputs, 1))
        self.hidden1 = np.matmul(self.weights_ih1, inputs) # feeding forward to hidden layer 1
        self.hidden1 = np.add(self.hidden1, self.bias_ih1) # adding the bias
        self.hidden1 = self.sigmoid(self.hidden1) # applying activation function 

        #       2) hidden1 --> hidden2
        self.hidden1 = np.asarray(self.hidden1)
        self.hidden1 = np.reshape(self.hidden1, (self.num_hiddenNodes1, 1))     # preparing self.hidden1 into the appropriate matrix form
        self.hidden2 = np.matmul(self.weights_h1h2, self.hidden1) # feeding forward to hidden layer 2
        self.hidden2 = np.add(self.hidden2, self.bias_h1h2) # adding the bias 
        self.hidden2 = self.sigmoid(self.hidden2) # applying the activation function

        #       3) hidden2--> output
        self.hidden2 = np.asarray(self.hidden2)
        self.hidden2 = np.reshape(self.hidden2, (self.num_hiddenNodes2, 1))     # preparing self.hidden2 into the appropriate matrix form
        self.outputs = np.matmul(self.weights_h2o, self.hidden2) # feeding forward to output layer
        self.outputs = np.add(self.outputs, self.bias_h2o) # adding the bias 
        self.outputs = self.sigmoid(self.outputs) # applying the activation function
        return self.outputs

    def train(self, inputs, targets, lr):
        # train the model
        self.outputs = self.feedforward(inputs)
        self.lr = lr
        
        # calculate error and delta w for:
        #   1) hidden2 --> output
        error_h2o = np.subtract(targets, self.outputs) # error
        gradient_h2o = self.dsigmoid(self.outputs)
        gradient_h2o = np.multiply(error_h2o, gradient_h2o)
        gradient_h2o = np.multiply(self.lr, gradient_h2o)
        delta_w_h2o = np.matmul(gradient_h2o, self.hidden2.transpose())

        #   2) hidden2 --> hidden1
        error_h1h2 = np.matmul(self.weights_h2o.transpose(), error_h2o)
        gradient_h1h2 = self.dsigmoid(self.hidden2)
        gradient_h1h2 = np.multiply(error_h1h2, gradient_h1h2)
        gradient_h1h2 = np.multiply(self.lr, gradient_h1h2)
        delta_w_h1h2 = np.matmul(gradient_h1h2, self.hidden1.transpose())

        #   3) hidden1 --> inputs
        error_ih1 = np.matmul(self.weights_h1h2.transpose(), error_h1h2)
        gradient_ih1 = self.dsigmoid(self.hidden1)
        gradient_ih1 = np.multiply(error_ih1, gradient_ih1)
        gradient_ih1 = np.multiply(self.lr, gradient_ih1)
        delta_w_ih1 = np.matmul(gradient_ih1, inputs.transpose())

        # adjust the weights and biases for:
        #    1) hidden2 --> output
        self.weights_h2o = np.add(self.weights_h2o, delta_w_h2o)
        self.bias_h2o = np.add(self.bias_h2o, gradient_h2o)

        #   2) hidden1 --> hidden2
        self.weights_h1h2 = np.add(self.weights_h1h2, delta_w_h1h2)
        self.bias_h1h2 = np.add(self.bias_h1h2, gradient_h1h2)

        #   3) input --> hidden1
        self.weights_ih1 = np.add(self.weights_ih1, delta_w_ih1)
        self.bias_ih1 = np.add(self.bias_ih1, gradient_ih1)
        
    def sigmoid(self, x):
        # helper function for activation sigmoid
        x = np.clip(x, -500, 500)
        function = 1 / (1 + (math.e ** (-x)))
        return function

    def dsigmoid(self, x):
        # helper function for finding gradient
        return x * (1 - x)

    def getOutput(self):
        # returns the output
        return self.outputs


def main():
    # define parameters  
    i = 784 # number of input nodes
    h1 = 1000 # number of nodes in hidden layer 1
    h2 = 1000 # number of nodes in hidden2
    o = 10 # number of output nodes
    lr = 0.001 # learning rate 

    nn = NeuralNetwork(i, h1, h2, o)
    
    # load training set
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # turn the data into an appropriate matrix form
    x_train = np.reshape(x_train, (60000, 784, 1))
    y_train = np.reshape(y_train, (60000, 1))

    targs = np.zeros((60000, 10, 1))
    for i in range(60000):
        targs[i][y_train[i][0]-1][0] = 1 # preparing the target matrix 
        
    
    # normalize data
    x_train = x_train / 255
    x_test = x_test / 255

    # train the network
    nn.setup()
    for epoch in range(5):
        for n in range(60000):
            ins = x_train[n]
            #targs = y_train[n]
            target = targs[n]
            nn.train(ins, target, lr)
    
    # testing with 10 test images 
    for i in range(len(x_test)-9990):
        test_image = x_test[i]
        plt.imshow(test_image)
        plt.show()
        output = nn.feedforward(test_image)
        print("The actual value is %d and the predicted value is %d" % (y_test[i], np.argmax(output)+1))

    
if __name__=='__main__':
    main()    
        
    
        
                
        

