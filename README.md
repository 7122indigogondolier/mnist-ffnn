# MNIST Feed-Forward Neural Network 

Implemented by Utkrist P Thapa,  
Washington and Lee University 

## Brief Description 
The Python code file mnist.py consists of a NeuralNetwork class. The class contains methods setup() feedforward() and train() along with other helper functions such as sigmoid() and dsigmoid(). The setup() method initializes the values for the weight matrices and feedforward() method feeds the input signal forward by multiplying the input with the weight matrices through different dense fully-connected layers. The train() method contains an implementation of the back-propagation of error along with gradient-descent for training the model. The main() module retrieves the mnist data, divides it into training and test sets (x_train and x_test) along with corresponding labels (y_train and y_test). 

There is also a Tensorflow implementation (tf-mnist.py) of the same thing with slight differences: 
  - The loss function is sparse_categorical_crossentropy
  - The optimizer used is adam
  - The number of hidden layers and the number of neurons in the hidden layer is different from mnist.py
  - Tensorflow has better under-the-hood implementation of its API calls that is most definitely not as simple as the   feedforward network implementation in mnist.py. Hence, the accuracy achieved using Tensorflow is greater than code built from scratch
  
## Data
The mnist dataset can be directly imported into the Python file (assuming Keras is installed) from keras.datasets. The dataset consists of 60,000 labeled training images each with 28 x 28 resolution. These images are handwritten digits ranging from 0 to 9. The test set consists of 10000 labeled images. 

## Hyperparameters in mnist.py: 
  - Epochs: 5
  - Learning rate: 0.1 
  - Number of hidden layers: 2
  - Number of neurons in the hidden layers: 1000 (each)
  
## Results 

**MNIST Handwritten Digits Recognition from Scratch**

I've used numpy arrays for matrices and numpy operations such as np.dot, np.matmul, np.add, np.subtract and so on for performing matrix operations in order to simulate a feedforward neural network. The helper functions also utilize the math library in Python.
I only used ten test cases and have included six here. The accuracy of the model has not been calculated. There were two incorrect predictions out of the ten that I observed. 

![mnist1](https://raw.githubusercontent.com/7122indigogondolier/mnist-ffnn/master/mnist1.png)
![mnist2](https://raw.githubusercontent.com/7122indigogondolier/mnist-ffnn/master/mnist2.png)
![mnist3](https://raw.githubusercontent.com/7122indigogondolier/mnist-ffnn/master/mnist3.png)

*The last prediction shown here is incorrect since a 100% accuracy wasn't achieved* 

**MNIST Handwritten Digits Recognition using Tensorflow**

I used Tensorflow's API calls in order to define the layers for the model, compile the model with the adam optimizer, sparse categorical crossentropy loss function, and accuracy as the evaluation metric. The picture shows a randomly selected test case and the prediction made by the model. 

![mnist4](https://raw.githubusercontent.com/7122indigogondolier/mnist-ffnn/master/mnist4.png)

*The code was run on Google Colab. Google Colab provides a useful platform for machine/deep learning enthusiasts who don't necessarily own their own GPUs.*

Author: Utkrist P. Thapa 



