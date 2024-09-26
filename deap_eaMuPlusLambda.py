import sys, os
import random
import time
from evoman.environment import Environment
from deap import creator, base, tools, algorithms
import numpy as np
from demo_controller import player_controller

# Log for visualization
def log(gen, ngen, population, stats, log_file, experiment_name):
    record = stats.compile(population)
    best_ind = tools.selBest(population, 1)[0]
    best_fitness = best_ind.fitness.values[0]

    gen_mean = record['avg']
    gen_std = np.std([ind.fitness.values[0] for ind in population])
    log_file.write(f"\n{gen} {best_fitness:.6f} {gen_mean:.6f} {gen_std:.6f}\n")
    
    if gen == 0 or gen == ngen:
        print(f"\n GENERATION {gen} - Best Ind: {best_ind[:5]}... - Experiment: {experiment_name}")
        np.savetxt(os.path.join(experiment_name, 'best.txt'), best_ind)

# Initialize DEAP
def init_deap(env):
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except AttributeError:
        pass

    n_hidden = 10
    n_weights = (env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_weights)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Gaussian mutation
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

# Fitness evaluation function
def evalOneMax(individual, env):
    individual_np = np.array(individual)
    f, p, e, t = env.play(pcont=individual_np)
    return f,

# Step 1: Initialize population
def init_pop(toolbox, mu):
    return toolbox.population(n=mu)

# Step 2: Evaluate population
def eval_pop(population, toolbox, env):
    fitnesses = list(map(lambda ind: toolbox.evaluate(ind, env), population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

# Step 3: Select best individuals among combined population and offspring
def selection(toolbox, population, offspring, mu):
    return toolbox.select(population + offspring, mu)

# Step 6: Replace population
def replace_pop(toolbox, parents, offspring, mu):
    return selection(toolbox, parents, offspring, mu)

# Main evolutionary loop
def run_muPlusLambda(env, mu=100, ngen=30, lambda_=100, cxpb=0.6, mutpb=0.2, experiment_name='dummy_demo_muPlusLambda'):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    toolbox = init_deap(env)
    population = init_pop(toolbox, mu)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    halloffame = tools.HallOfFame(1)

    # Initialize log
    with open(os.path.join(experiment_name, 'results.txt'), 'a') as log_file:
        log_file.write("\ngen best mean std\n")
        start = time.time()

        # Step 2: Evaluate the initial population
        eval_pop(population, toolbox, env)
        halloffame.update(population)
        log(0, ngen, population, stats, log_file, experiment_name)

        # Evolve over generations
        for gen in range(1, ngen + 1):
            # Generate offspring
            offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

            # Evaluate offspring
            eval_pop(offspring, toolbox, env)

            # Replacement: Select next generation from parents + offspring
            population = replace_pop(toolbox, population, offspring, mu)
            halloffame.update(population)

            log(gen, ngen, population, stats, log_file, experiment_name)

        end = time.time()  # End timer
        print(f"\nExecution time: {round((end - start) / 60, 2)} minutes \n")
    
    env.state_to_log()
