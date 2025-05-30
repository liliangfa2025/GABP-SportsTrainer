import numpy as np

class GAOptimizer:
    def __init__(self, bp_network, params):
        self.bp = bp_network
        self.population_size = params.get('population_size', 50)
        self.mutation_rate = params.get('mutation_rate', 0.1)
        self.generations = params.get('generations', 100)

    def evolve(self, X_train, y_train, max_epochs=100):
        population = [self._random_weights() for _ in range(self.population_size)]
        for generation in range(self.generations):
            fitness_scores = [self._fitness(individual, X_train, y_train) for individual in population]
            top_indices = np.argsort(fitness_scores)[:self.population_size // 2]
            selected = [population[i] for i in top_indices]
            offspring = self._crossover(selected)
            population = selected + self._mutate(offspring)
            if generation % 10 == 0:
                print(f"Generation {generation}, Best MSE: {min(fitness_scores)}")
        best = population[np.argmin(fitness_scores)]
        self._set_weights(best)

    def _fitness(self, weights, X, y):
        self._set_weights(weights)
        preds = self.bp.forward(X)
        return np.mean((preds - y) ** 2)

    def _random_weights(self):
        return np.concatenate([self.bp.w1.flatten(), self.bp.b1.flatten(), self.bp.w2.flatten(), self.bp.b2.flatten()])

    def _set_weights(self, weights):
        idx = 0
        shapes = [self.bp.w1.shape, self.bp.b1.shape, self.bp.w2.shape, self.bp.b2.shape]
        for param, shape in zip([self.bp.w1, self.bp.b1, self.bp.w2, self.bp.b2], shapes):
            size = np.prod(shape)
            param[:] = weights[idx:idx+size].reshape(shape)
            idx += size

    def _crossover(self, selected):
        offspring = []
        for i in range(len(selected) - 1):
            parent1 = selected[i]
            parent2 = selected[i+1]
            cut = np.random.randint(1, len(parent1)-1)
            child = np.concatenate([parent1[:cut], parent2[cut:]])
            offspring.append(child)
        return offspring

    def _mutate(self, population):
        for i in range(len(population)):
            if np.random.rand() < self.mutation_rate:
                idx = np.random.randint(len(population[i]))
                population[i][idx] += np.random.randn() * 0.1
        return population
