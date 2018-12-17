import numpy as np


class Network(object):
    def __init__(self):
        self._layers = []

    def validate(self):
        pass

    def forward(self, input):
        out = input
        for layer in self._layers:
            out = layer.forward(out)
        return out

    def backward(self):
        pass


class Layer(object):
    def __init__(self, width, output):
        self.width = width
        self.output = output
        self.activation = None
        self.weights = np.array([width,output])
        self.bias = np.array([width])

    def forward(self, idata):
        h = idata.T.dot(self.weights)
        return h

    def backward(self):
        pass


class Model(object):
    def __init__(self):
        pass

    def train(self, input):
        pass

    def evaluate(self):
        pass

    def report(self):
        pass


if __name__ == "__main__":
    m = Model()
    m.train()
    m.evaluate()
