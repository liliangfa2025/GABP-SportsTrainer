import numpy as np
import yaml
from models.gabp_model import GABPNetwork
from models.utils import mean_squared_error

# Load configuration
with open('src/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Simulate test data (replace with your test data)
np.random.seed(42)
X_test = np.random.rand(100, config['model']['input_size'])
y_test = np.random.rand(100, config['model']['output_size'])

# Initialize GABP model (replace with loading a trained model if available)
gabp = GABPNetwork(
    input_size=config['model']['input_size'],
    hidden_size=config['model']['hidden_size'],
    output_size=config['model']['output_size'],
    ga_params=config['ga']
)

# Run prediction
y_pred = gabp.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")
