import numpy as np

class BPNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        d_a2 = (self.a2 - y) * self.sigmoid_derivative(self.a2)
        d_w2 = np.dot(self.a1.T, d_a2) / m
        d_b2 = np.mean(d_a2, axis=0, keepdims=True)

        d_a1 = np.dot(d_a2, self.w2.T) * self.sigmoid_derivative(self.a1)
        d_w1 = np.dot(X.T, d_a1) / m
        d_b1 = np.mean(d_a1, axis=0, keepdims=True)

        self.w2 -= self.lr * d_w2
        self.b2 -= self.lr * d_b2
        self.w1 -= self.lr * d_w1
        self.b1 -= self.lr * d_b1

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
