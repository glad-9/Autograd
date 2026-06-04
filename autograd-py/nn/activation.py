from core.tensor import Tensor

class ReLU:
    def forward(self, X):
        return X.relu()

    def get_params(self):
        return []

class Sigmoid:
    def forward(self, X):
        return X.sigmoid()

    def get_params(self):
        return []
