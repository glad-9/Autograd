import numpy as np
from core.tensor import Tensor

class Linear:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.W = Tensor(np.random.rand(self.input_size, self.output_size), requires_grad=True)
        self.b = Tensor(np.zeros((1, self.output_size)), requires_grad=True)

    def forward(self, X):
        X = X if isinstance(X, Tensor) else Tensor(np.array(X))
        Z = X @ self.W + self.b
        return Z

    def get_params(self):
        return [self.W, self.b]