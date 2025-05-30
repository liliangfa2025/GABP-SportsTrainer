import numpy as np
import yaml
from models.gabp_model import GABPNetwork

# Load configuration
with open('src/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Example new input for inference (replace with your real data)
new_input = np.random.rand(1, config['model']['input_size'])

# Initialize GABP model (replace with loading a trained model if available)
gabp = GABPNetwork(
    input_size=config['model']['input_size'],
    hidden_size=config['model']['hidden_size'],
    output_size=config['model']['output_size'],
    ga_params=config['ga']
)

# Make prediction
output = gabp.predict(new_input)
print("Predicted output:", output)
