import numpy as np
import unittest
from models.utils import normalize, train_test_split, mean_squared_error

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [3, 4], [5, 6]])
        self.y_true = np.array([1, 2, 3])
        self.y_pred = np.array([1.1, 1.9, 3.2])

    def test_normalize(self):
        norm = normalize(self.X)
        self.assertTrue(np.all(norm >= 0) and np.all(norm <= 1))

    def test_train_test_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y_true, test_ratio=0.33)
        self.assertEqual(X_train.shape[0] + X_test.shape[0], self.X.shape[0])

    def test_mse(self):
        mse = mean_squared_error(self.y_true, self.y_pred)
        self.assertTrue(mse >= 0)

if __name__ == '__main__':
    unittest.main()
