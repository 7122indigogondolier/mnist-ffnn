# MNIST Feed-Forward Neural Network 

Implemented by Utkrist P Thapa 
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

Author: Utkrist P. Thapa 



