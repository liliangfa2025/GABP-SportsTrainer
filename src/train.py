import numpy as np
import yaml
from models.gabp_model import GABPNetwork
from models.utils import train_test_split, normalize

# Load configuration
with open('src/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Simulate or load your dataset (replace this with actual data loading)
np.random.seed(42)
X = np.random.rand(500, config['model']['input_size'])
y = np.random.rand(500, config['model']['output_size'])

# Normalize and split data
X = normalize(X)
y = normalize(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

# Initialize GABP model
gabp = GABPNetwork(
    input_size=config['model']['input_size'],
    hidden_size=config['model']['hidden_size'],
    output_size=config['model']['output_size'],
    ga_params=config['ga']
)

# Train the model
gabp.train(X_train, y_train, max_epochs=config['train']['epochs'])

print("Training completed.")
