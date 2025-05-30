# 🏋️‍♂️ GABP-Based Sports Training Effect Research

## 📚 Overview

This project investigates the application of **Generalized Adaptive Backpropagation (GABP)** neural networks—an AI model combining Genetic Algorithms (GA) with Backpropagation (BP)—to optimize and evaluate sports training effectiveness.  
The goal is to build an intelligent and personalized training system for athletes, leveraging biometric signals, performance data, and training logs for adaptive learning.

The repository contains the core model implementations, utility functions, training scripts, evaluation pipelines, and unit tests to support reproducible research and further development.

---

## 🏗️ Project Structure

GABP-Sports-Training/ │ ├── data/ # Raw and processed data for training and evaluation ├── models/ # Core models (GABP, BP baseline, GA utilities) ├── results/ # Experiment results (plots, reports) ├── src/ # Training, evaluation, inference scripts and config ├── tests/ # Unit tests for models and utilities ├── .gitignore # Files to ignore in version control ├── LICENSE # License file (e.g., MIT) ├── README.md # This file ├── requirements.txt # Python dependencies └── setup.py # (Optional) Package setup script


---

## 🚀 Quick Start

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
2️⃣ Configure hyperparameters
Edit the file src/config.yaml to adjust:
Model input, hidden, and output sizes
Training epochs
Genetic Algorithm parameters (population size, mutation rate, generations)
3️⃣ Train the model
python src/train.py
4️⃣ Evaluate the model
python src/evaluate.py
5️⃣ Run inference
python src/inference.py



📦 Features
✅ GABP Neural Network: Combines Genetic Algorithms with Backpropagation for improved convergence and accuracy.
✅ Baseline BP Model: For comparative experiments.
✅ Data Handling Utilities: For normalization, splitting, and error metrics.
✅ Unit Tests: Ensure core functions are working as expected.
✅ Configurable: Modify hyperparameters via config.yaml.


🧪 Testing
Run all tests:
python -m unittest discover -s tests

Or run individual test files:
python tests/test_models.py
python tests/test_utils.py

📊 Experimental Results
Results and visualizations (e.g., MSE, learning curves) are saved in the results/ directory after experiments.

📄 License
This project is licensed under the MIT License. See LICENSE for details.

🤝 Contributions
Contributions, improvements, and feedback are welcome!
Feel free to open issues or submit pull requests.
