import numpy as np


class Value():
    def __init__(self, data):
        self.data = data
        self.grad = 0
        self._prev = 0
        self._backward = lambda: None

    def __add__(self, other):
        out = Value(self.data + other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += out.grad * 1
            other.grad += out.grad * 1

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        return out

    def backward():
        ...
