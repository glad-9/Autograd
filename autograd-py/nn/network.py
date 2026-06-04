from core.tensor import Tensor

class Network:
    def __init__(self, layers, loss, optimizer):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        
    def forward_prop(self, X):
        activation = X
        for layer in self.layers:
            activation = layer.forward(activation)

        return activation

    def compute_loss(self, X, y):
        y_hat = self.forward_prop(X)
        return self.loss.forward(y_hat, y) 

    def backward_prop(self, X, y):
        loss = self.compute_loss(X, y)
        loss.backward()
        return loss.data.item()

    def fit(self, train, lr, iterations):
        loss_history = []
        X, y = train

        for i in range(iterations):
            loss = self.backward_prop(X, y)
            self.optimizer.step(self.layers, lr)
            loss_history.append(loss)

            if i % 100 == 0:
                print(f"Epoch: {i / 100}\nLoss:{loss}")

        return loss_history[-1]


            



