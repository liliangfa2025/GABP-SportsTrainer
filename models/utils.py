import numpy as np

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def train_test_split(X, y, test_ratio=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    test_size = int(X.shape[0] * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
