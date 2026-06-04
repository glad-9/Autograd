from core.tensor import Tensor
from nn.activation import ReLU, Sigmoid
from nn.optimizer import SGD
from nn.loss import BCE
from nn.layer import Linear
from nn.network import Network


loss = BCE()
optimizer = SGD()

model = Network(
    [Linear(2, 4), ReLU(), Linear(4, 2), ReLU(), Linear(2, 1), Sigmoid()],
    loss,
    optimizer,
)

