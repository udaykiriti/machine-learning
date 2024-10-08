from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data = load_breast_cancer()
X, Y = data.data, data.target

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Genetic algorithm parameters
population_size = 20
num_generations = 50
mutation_rate = 0.1

# Initialize population
population = np.random.randint(2, size=(population_size, X_train.shape[1]))

def fitness_function(chromosome):
    selected_features = np.where(chromosome == 1)[0]
    if len(selected_features) == 0:
        return 0
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    model = KNeighborsClassifier()
    model.fit(X_train_selected, Y_train)
    Y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(Y_test, Y_pred)
    return accuracy

def select_mating_pool(population, fitness_scores, num_mating):
    selected_indices = np.argsort(fitness_scores)[-num_mating:]
    return population[selected_indices]

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2

def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# Genetic algorithm
for generation in range(num_generations):
    fitness_scores = np.array([fitness_function(chromosome) for chromosome in population])
    mating_pool = select_mating_pool(population, fitness_scores, population_size // 2)

    new_population = []
    while len(new_population) < population_size:
        parents = np.random.choice(mating_pool.shape[0], size=2, replace=False)
        child1, child2 = crossover(mating_pool[parents[0]], mating_pool[parents[1]])
        new_population.append(mutate(child1, mutation_rate))
        new_population.append(mutate(child2, mutation_rate))

    population = np.array(new_population)

# Get the best chromosome
fitness_scores = np.array([fitness_function(chromosome) for chromosome in population])
best_chromosome = population[np.argmax(fitness_scores)]
print("Best Feature Set:", best_chromosome)
print("Best Fitness Score:", max(fitness_scores))
