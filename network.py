import numpy as np
from random import random

weights_table = {}
bias_table = {}
grad_weights_table = {}
grad_bias_table = {}

class Layer(object):
    """Represent a layer in Neuron Network"""
    def __init__(self, input_width, width, activation=None):
        self.width = width
        self.input_width = input_width
        self.activation = activation
        # weighted input
        self.h = None
        # after activation function
        self.output = None
        self.grad_input = None
        self.output = None
        self.index = None
        self.is_last_layer = False

    @property
    def weights(self):
        return weights_table.get(self.index)

    @property
    def bias(self):
        return bias_table.get(self.index)

    def initialize(self):
        weights_table[self.index] = np.random.rand(self.width, self.input_width)
        bias_table[self.index] = np.random.rand(self.width, 1)

    def debug_print(self):
        print("Weights", self.weights)
        print("output", self.output)

    def forward(self, input_data):
        # weighted input
        self.h = np.dot(input_data, self.weights) + self.bias
        (func, dev) = self.get_activation()
        if func:
            self.output = func(self.h)
        else:
            self.output = self.h
        self.debug_print()

    @staticmethod
    def activation_func1(x):
        return 1. / (1 + np.exp(-x))
    
    @staticmethod
    def activation_dev1(output):
        return output*(1-output)
    
    @staticmethod
    def activation_func2(x):
        return max(x, 0)
        
    @staticmethod
    def activation_dev2(output):
        return np.sgn(output)
        
    def get_activation(self)
        if self.activation == "sigmoid":
            return (self.activation_func1, self.activation_dev1)
        elif self.activation == "relu":
            return (self.activation_func2, self.activation_dev2)
        else
            return (None, None)
          
    def calculate_gradient(self, prev_grad):
        (func, dev) = self.get_activation()
        
        shared_grad = prev_grad
        if not func:
            shared_grad = shared_grad * dev(self.ouput)
        nr_of_samples = prev_grad.shape[0]
        grad_weight = 1/nr_of_sample * np.sum(np.outer(shared_grad, self.prev_layer.output), axis=0)
        grad_weights_table[self.index] = grad_weight
        transfer_grad = np.dot(shared_grad, self.weight.T)
        
        return transfer_grad

    def update(self, learning_rate=0.1):
        self.weights -= learning_rate * self.grad_weight


class Model(object):
    def __init__(self):
        self.input_layer = Layer(2, 1)
        self._layers = []
        self._layers.append(Layer(2, 2))
        self._layers[-1].prev_layer = self.input_layer
        self._layers.append(Layer(1, 2))
        self._layers[-1].prev_layer = self._layers[-2]

    def train(self):
        X = np.array([[1, 1], [0, 0], [0, 1], [1,0]])
        Y = np.array([[0], [0], [1], [1]])

        output = self.evaluate(X)
        self.backprop(self.grad_output(output, Y))
        self.update()

    def grad_output(self, output, y):
        return y-output

    def evaluate(self, input_data):
        self.input_layer.output = input_data
        for layer in self._layers:
            layer.forward()
        return self._layers[-1].output

    def backprop(self, prev_grad):
        for layer in reversed(self._layers):
            prev_grad = layer.calculate_gradient(prev_grad)

    def update(self):
        for layer in self._layers:
            layer.update(self.learning_rate)

    def initialize(self):
        """initialize weights and biases"""
        for layer in self._layers:
            layer.initialize()

    def report(self):
        pass


if __name__ == "__main__":
    m = Model(1)
    m.train()
    out = m.evaluate([1, 1])
    print(out)
