import os
import random
import time
from a2.utils import fitness
from deap import creator, base, tools, algorithms
import numpy as np


# Log for visualization
def log(gen, ngen, population, stats, log_file, experiment_name):
    record = stats.compile(population)
    best_ind = tools.selBest(population, 1)[0]
    best_fitness = best_ind.fitness.values[0]

    gen_mean = record["avg"]
    gen_std = np.std([ind.fitness.values[0] for ind in population])
    log_file.write(f"\n{gen} {best_fitness:.6f} {gen_mean:.6f} {gen_std:.6f}\n")

    if gen == 0 or gen == ngen:
        np.savetxt(os.path.join(experiment_name, "best.txt"), best_ind)


# Initialize DEAP
def init_deap(env):
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    n_hidden = 10
    n_weights = (env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=n_weights,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
    toolbox.register(
        "mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2
    )  # Gaussian mutation
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


def evalOneMax(individual, env):
    return fitness(individual, env)


# Step 1: Initialize population
def init_pop(toolbox, npop):
    return toolbox.population(n=npop)


# Step 2: Evaluate population
def eval_pop(population, toolbox, env):
    fitnesses = list(map(lambda ind: toolbox.evaluate(ind, env), population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit


# Step 3: Select best individuals among combined population and offspring
def selection(toolbox, population, offspring, npop):
    return toolbox.select(population + offspring, npop)


# Step 6: Replace population
def replace_pop(toolbox, parents, offspring, npop):
    return selection(toolbox, parents, offspring, npop)


# Main evolutionary loop
def run_muPlusLambda_generalist(
    env,
    train,
    npop,
    ngen,
    lambda_,
    cxpb,
    mutpb,
    experiment_name,
):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    toolbox = init_deap(env)
    population = init_pop(toolbox, npop)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    halloffame = tools.HallOfFame(1)
    found_max_fitness = None

    # Initialize log
    with open(os.path.join(experiment_name, "results.txt"), "a") as log_file:
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
            population = replace_pop(toolbox, population, offspring, npop)
            halloffame.update(population)

            log(gen, ngen, population, stats, log_file, experiment_name)

            best_ind = tools.selBest(population, 1)[0]
            found_max_fitness = best_ind.fitness.values[0]

            train.report({"fitness": found_max_fitness})

        end = time.time()  # End timer
        # print(f"\nExecution time: {round((end - start) / 60, 2)} minutes \n")

    env.state_to_log()

    return found_max_fitness, halloffame[0]
