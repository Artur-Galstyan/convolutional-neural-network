import numpy as np

import datahandler


def init_layer(n_in, n_out) -> tuple[np.ndarray, np.ndarray]:
    w = np.random.randn(n_in, n_out) * 0.1
    b = (
        np.random.randn(
            n_out,
        )
        * 0.1
    )

    return w, b


def forward_single_dense_layer(x: np.ndarray, w: np.ndarray, b: np.ndarray):
    return x @ w + b


def get_weight_gradient_single_dense_layer(x: np.ndarray, dE_dY: np.ndarray):
    return x.T @ dE_dY


def get_bias_gradient_single_dense_layer(dE_dY: np.ndarray):
    return np.sum(dE_dY, axis=0) / dE_dY.shape[0]


def get_error_for_previous_layer(dE_dY: np.ndarray, w: np.ndarray):
    return dE_dY @ w.T


class Dense:
    def __init__(self, n_in, n_out) -> None:
        self.w, self.b = init_layer(n_in, n_out)

    def forward(self, x):
        self.x = x
        return forward_single_dense_layer(x, self.w, self.b)

    def backward(self, dE_dY):
        dW = get_weight_gradient_single_dense_layer(self.x, dE_dY)
        dB = get_bias_gradient_single_dense_layer(dE_dY)
        dX = get_error_for_previous_layer(dE_dY, self.w)
        return dW, dB, dX

    def update(self, dW, dB, learning_rate):
        self.w -= learning_rate * dW
        self.b -= learning_rate * dB


class ReLU:
    def __init__(self) -> None:
        self.x = None

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_backward(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def forward(self, x):
        self.x = x
        return self.relu(x)

    def backward(self, dE_dY):
        dX = np.multiply(dE_dY, self.relu_backward(self.x))
        return 0, 0, dX

    def update(self, dW, dB, learning_rate):
        pass


class Sigmoid:
    def __init__(self):
        self.x = None
        pass

    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def sigmoid_backward(x):
        sig = Sigmoid.sigmoid(x)
        return sig * (1 - sig)

    def forward(self, x):
        self.x = x
        return self.sigmoid(x)

    def backward(self, dE_dY):
        dX = np.multiply(dE_dY, self.sigmoid_backward(self.x))
        return 0, 0, dX

    def update(self, dW, dB, learning_rate):
        pass


class Network:
    def __init__(self, layers) -> None:
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dE_dY):
        grads = []
        for layer in reversed(self.layers):
            dW, dB, dX = layer.backward(dE_dY)
            grads.append((dW, dB))
            dE_dY = dX

        return reversed(grads)

    def update(self, learning_rate, grads):
        for layer, grad in zip(self.layers, grads):
            dW, dB = grad
            layer.update(dW, dB, learning_rate)


def get_current_accuracy(network, X_test, y_test):
    correct = 0
    total_counter = 0
    for x, y in zip(X_test, y_test):
        a = network.forward(x)
        pred = np.argmax(a, axis=1, keepdims=True)
        y = np.argmax(y, axis=1, keepdims=True)
        correct += (pred == y).sum()
        total_counter += len(x)
    accuracy = correct / total_counter
    return accuracy


def mse_loss(prediction, target):
    return 2 * (prediction - target) / np.size(prediction)


def main():
    X_train, y_train, X_test, y_test = datahandler.get_mnist()
    network = Network(
        [
            Dense(784, 64),
            ReLU(),
            Dense(64, 32),
            ReLU(),
            Dense(32, 16),
            ReLU(),
            Dense(16, 10),
            Sigmoid(),
        ]
    )

    n_epochs = 50
    learning_rate = 0.1
    for epoch in range(n_epochs):
        for x, y in zip(X_train, y_train):
            a = network.forward(x)
            error = mse_loss(a, y)
            grads = network.backward(error)
            network.update(learning_rate, grads)
        accuracy = get_current_accuracy(network, X_test, y_test)
        print(f"Epoch {epoch} Accuracy = {np.round(accuracy * 100, 2)}%")


if __name__ == "__main__":
    main()
