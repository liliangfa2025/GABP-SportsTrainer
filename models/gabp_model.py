import numpy as np
from ga_utils import GAOptimizer
from bp_baseline import BPNetwork

class GABPNetwork:
    def __init__(self, input_size, hidden_size, output_size, ga_params):
        self.bp = BPNetwork(input_size, hidden_size, output_size)
        self.ga = GAOptimizer(self.bp, ga_params)

    def train(self, X_train, y_train, max_epochs=1000):
        self.ga.evolve(X_train, y_train, max_epochs)

    def predict(self, X):
        return self.bp.forward(X)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        mse = np.mean((preds - y_test) ** 2)
        return mse
