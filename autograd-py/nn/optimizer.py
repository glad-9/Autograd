import numpy as np
from core.tensor import Tensor

class SGD:
    name = "sgd"

    def step(self, layers, lr):
        for layer in layers:
            for p in layer.get_params():
                p.data -= lr * p.grad
                p.grad = None