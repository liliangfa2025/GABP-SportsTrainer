model:
  input_size: 10          # Input dimension (number of features)
  hidden_size: 20         # Number of neurons in the hidden layer
  output_size: 1          # Output dimension (e.g., for regression tasks)

train:
  epochs: 100             # Number of training epochs

ga:
  population_size: 50     # Size of the genetic algorithm population
  mutation_rate: 0.1      # Mutation rate for GA
  generations: 50         # Number of generations for evolution
