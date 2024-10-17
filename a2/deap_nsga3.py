# imports framework
import math
import os
import random
import time
from deap import creator, base, tools, algorithms
import numpy as np


# log for visualization
def log(gen, ngen, population, stats, log_file, experiment_name):
    record = stats.compile(population)
    best_ind = tools.selBest(population, 1)[0]
    best_fitness = best_ind.fitness.values[0]  # Average fitness across multiple enemies

    gen_mean = record["avg"]
    gen_std = np.std([ind.fitness.values[0] for ind in population])

    log_file.write(f"\n{gen} {best_fitness:.6f} {gen_mean:.6f} {gen_std:.6f}\n")

    if gen == 0 or gen == ngen:
        np.savetxt(os.path.join(experiment_name, "best.txt"), best_ind)


def evalOneMax(individual, env):
    individual_np = np.array(individual)
    f, p, e, t = env.play(pcont=individual_np)

    fitness = 0.9 * (100 - e) - math.pow(math.e, (100 - p) / 20) - np.log(t)

    return (fitness, p, e, t)


# initialize DEAP
def init_deap(env):
    # NN paramtetrs
    n_hidden = 10
    n_weights = (env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
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
    toolbox.register("mate", tools.cxTwoPoint)  # two-point crossover
    toolbox.register(
        "mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2
    )  # gaussian mutation
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


# Step 1: Initialize population
def init_pop(toolbox, npop):
    return toolbox.population(n=npop)


# Step 2: Evaluate population using evaluation function (evalOneMax)
def eval_pop(individuals, toolbox, env):
    """Evaluate the fitness of a list of individuals."""
    fitnesses = list(map(lambda ind: toolbox.evaluate(ind, env), individuals))
    for ind, stats in zip(individuals, fitnesses):
        (fitness, p, e, t) = stats
        ind.fitness.values = (fitness,)  # Store fitness
        ind.p = p  # Store player health
        ind.e = e  # Store enemy health
        ind.t = t  # Store time


# Step 3: Select the next generation individuals
def selection(toolbox, population):
    return toolbox.select(population, len(population))


# Step 6: Replace the old population with new offspring
def replacement(population, offspring):
    # print(f"Replacing population with new offspring....")
    population[:] = offspring


def run_nsga3_generalist(
    env,
    report,
    npop,
    ngen,
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

        # Evaluate initial population
        eval_pop(population, toolbox, env)

        if halloffame is not None:
            halloffame.update(population)

        if log_file:
            log(0, ngen, population, stats, log_file, experiment_name)

            # Evolve

        for gen in range(1, ngen + 1):
            offspring = selection(toolbox, population)
            offspring = list(map(toolbox.clone, offspring))

            # Crossover and mutation (DEAP varAnd)
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

            # Find individuals with invalid fitness and evaluate
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            eval_pop(invalid_ind, toolbox, env)

            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the old population with the new offspring
            replacement(population, offspring)

            if log_file:
                log(gen, ngen, population, stats, log_file, experiment_name)

            best_ind = tools.selBest(population, 1)[0]
            found_max_fitness = best_ind.fitness.values[0]
            p, e, t = best_ind.p, best_ind.e, best_ind.t
            print(
                f"Generation {gen}: Best fitness: {found_max_fitness}, Player health: {p}, Enemy health: {e}, Time: {t}"
            )
            report({"gain": p - e, "fitness": found_max_fitness})

        end = time.time()  # End timer

    env.state_to_log()

    return found_max_fitness, halloffame[0]
