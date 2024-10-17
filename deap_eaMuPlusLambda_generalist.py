import math
import sys, os
import random
import time
from evoman.environment import Environment
from deap import creator, base, tools, algorithms
import numpy as np
from demo_controller import player_controller
from tqdm import tqdm


# Log for visualization
def log(gen, ngen, population, stats, log_file, experiment_name):
    record = stats.compile(population)
    best_ind = tools.selBest(population, 1)[0]
    best_fitness = best_ind.fitness.values[0]

    gen_mean = record["avg"]
    gen_std = np.std([ind.fitness.values[0] for ind in population])
    log_file.write(f"\n{gen} {best_fitness:.6f} {gen_mean:.6f} {gen_std:.6f}\n")

    if gen == 0 or gen == ngen:
        # print(
        #     f"\n GENERATION {gen} - Best Ind: {best_ind[:5]}... - Experiment: {experiment_name}"
        # )
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


# Fitness evaluation function
# def evalOneMax(individual, env):
#     individual_np = np.array(individual)
#     total_fitness = 0
#     for enemy in env.enemies:  # Loop through all enemies in the group
#         env.update_parameter(
#             "enemies", [enemy]
#         )  # Temporarily set environment to this enemy
#         f, p, e, t = env.play(pcont=individual_np)
#         total_fitness += f  # Accumulate fitness from each enemy
#     avg_fitness = total_fitness / len(env.enemies)  # Average fitness across all enemies
#     return (avg_fitness,)


def evalOneMax(individual, env):
    individual_np = np.array(individual)
    f, p, e, t = env.play(pcont=individual_np)

    fitness = 0.9 * (100 - e) - math.pow(math.e, (100 - p) / 20) - np.log(t)

    return (fitness,)


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


def run_generation(
    toolbox,
    population,
    mu,
    env,
    halloffame,
    gen,
    ngen,
    stats,
    log_file,
    experiment_name,
    lambda_,
    cxpb,
    mutpb,
):
    # Generate offspring
    offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

    # Evaluate offspring
    eval_pop(offspring, toolbox, env)

    # Replacement: Select next generation from parents + offspring
    population = replace_pop(toolbox, population, offspring, mu)
    halloffame.update(population)

    log(gen, ngen, population, stats, log_file, experiment_name)


# Main evolutionary loop
def run_muPlusLambda_generalist(
    env,
    mu=200,
    ngen=30,
    lambda_=100,
    cxpb=0.69,
    mutpb=0.1,
    experiment_name="dummy_demo_muPlusLambda",
    run_num=None,
):
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
    with open(os.path.join(experiment_name, "results.txt"), "a") as log_file:
        log_file.write("\ngen best mean std\n")
        start = time.time()

        # Step 2: Evaluate the initial population
        eval_pop(population, toolbox, env)
        halloffame.update(population)
        log(0, ngen, population, stats, log_file, experiment_name)

        # Evolve over generations
        if run_num is not None:
            with tqdm(total=ngen, position=run_num) as progress:
                for gen in range(1, ngen + 1):
                    run_generation(
                        toolbox,
                        population,
                        mu,
                        env,
                        halloffame,
                        gen,
                        ngen,
                        stats,
                        log_file,
                        experiment_name,
                        lambda_,
                        cxpb,
                        mutpb,
                    )

                    progress.update(1)
        else:
            for gen in range(1, ngen + 1):
                run_generation(
                    toolbox,
                    population,
                    mu,
                    env,
                    halloffame,
                    gen,
                    ngen,
                    stats,
                    log_file,
                    experiment_name,
                    lambda_,
                    cxpb,
                    mutpb,
                )

        end = time.time()  # End timer
        # print(f"\nExecution time: {round((end - start) / 60, 2)} minutes \n")

    env.state_to_log()
