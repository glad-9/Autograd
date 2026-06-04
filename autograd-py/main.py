import numpy as np

from core.tensor import Tensor
from nn.activation import ReLU, Sigmoid
from nn.optimizer import SGD
from nn.loss import BCE
from nn.layer import Linear
from nn.network import Network

def generate_xor_continuous(num_samples=1000, noise=0.1):
    # 1. Generate uniform continuous random points between -1 and 1
    X = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, 2))
    
    # 2. Assign classes based on quadrant criteria:
    # Quadrant 1 (+,+) & 3 (-,-) -> XOR = 0
    # Quadrant 2 (-,+) & 4 (+,-) -> XOR = 1
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    
    # 3. Optional: Inject noise by flipping random labels
    if noise > 0:
        flip_mask = np.random.rand(num_samples) < noise
        y[flip_mask] = 1 - y[flip_mask]
        
    return X, y.reshape(-1, 1)

X_train, y_train = generate_xor_continuous(500, noise=0.05)
train = (Tensor(X_train), Tensor(y_train))

loss = BCE()
optimizer = SGD()

model = Network(
    [Linear(2, 4), ReLU(), Linear(4, 2), ReLU(), Linear(2, 1), Sigmoid()],
    loss,
    optimizer,
)

final_cost = model.fit(train=train, lr=0.01, iterations=10000)

