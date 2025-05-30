import numpy as np
import unittest
from models.bp_baseline import BPNetwork
from models.gabp_model import GABPNetwork
from models.ga_utils import GAOptimizer

class TestBPNetwork(unittest.TestCase):
    def setUp(self):
        self.model = BPNetwork(input_size=5, hidden_size=10, output_size=1)
        self.X = np.random.rand(20, 5)
        self.y = np.random.rand(20, 1)

    def test_forward_shape(self):
        output = self.model.forward(self.X)
        self.assertEqual(output.shape, (20, 1))

    def test_backward_no_error(self):
        try:
            self.model.forward(self.X)
            self.model.backward(self.X, self.y)
        except Exception as e:
            self.fail(f"Backward pass failed with error: {e}")

class TestGABPNetwork(unittest.TestCase):
    def setUp(self):
        ga_params = {'population_size': 10, 'mutation_rate': 0.1, 'generations': 5}
        self.model = GABPNetwork(input_size=5, hidden_size=10, output_size=1, ga_params=ga_params)
        self.X = np.random.rand(20, 5)
        self.y = np.random.rand(20, 1)

    def test_train_and_predict(self):
        self.model.train(self.X, self.y, max_epochs=10)
        preds = self.model.predict(self.X)
        self.assertEqual(preds.shape, (20, 1))

if __name__ == '__main__':
    unittest.main()
