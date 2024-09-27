# imports framework
import sys, os
import random
import time
from deap import creator, base, tools, algorithms
import numpy as np


#log for visualization
def log(gen, ngen, population, stats, log_file, experiment_name):
    record = stats.compile(population)
    best_ind = tools.selBest(population, 1)[0]
    best_fitness = best_ind.fitness.values[0]

    gen_mean = record['avg']
    gen_std = np.std([ind.fitness.values[0] for ind in population])
    # print(f"\n GENERATION {gen} - Best Fitness: {best_fitness:.6f}, Mean Fitness: {gen_mean:.6f}, Std Dev: {gen_std:.6f}")
    log_file.write(f"\n{gen} {best_fitness:.6f} {gen_mean:.6f} {gen_std:.6f}\n")
    
    if gen == 0 or gen == ngen:
        print(f"\n GENERATION {gen} - Best Ind: {best_ind[:5]}... - Experiment: {experiment_name}")
        np.savetxt(os.path.join(experiment_name, 'best.txt'), best_ind)


#fitness evaluation function
def evalOneMax(individual, env):
    individual_np = np.array(individual)
    f, p, e, t = env.play(pcont=individual_np)
    return f,

#initialize DEAP
def init_deap(env):
    # NN paramtetrs
    n_hidden = 10
    n_weights = (env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_weights)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)  # two-point crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # gaussian mutation
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


# Step 1: Initialize population
def init_pop(toolbox, npop):
    return toolbox.population(n=npop)

# Step 2: Evaluate population using evaluation function (evalOneMax)
def eval_pop(individuals, env):
    """Evaluate the fitness of a list of individuals."""
    # print(f"Evaluating population of {len(individuals)} individuals.")
    fitnesses = list(map(lambda ind: evalOneMax(ind, env), individuals))
    for ind, fit in zip(individuals, fitnesses):
        ind.fitness.values = fit
        # print(f"Individual Fitness: {fit[0]:.6f}")

# Step 3: Select the next generation individuals
def selection(toolbox, population):
    return toolbox.select(population, len(population))

# Step 6: Replace the old population with new offspring
def replacement(population, offspring):
    # print(f"Replacing population with new offspring....")
    population[:] = offspring

def eaSimple(population, toolbox, cxpb, mutpb, ngen, env, stats=None, halloffame=None, verbose=False, log_file=None, experiment_name='dummy_demo_eaSimple'):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate initial population
    eval_pop(population, env)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    if log_file:
        log(0, ngen, population, stats, log_file, experiment_name)

    # Evolve
    for gen in range(1, ngen + 1):
        # print(f"\n=== Generation {gen} ===")
        
        # Select the next generation individuals
        offspring = selection(toolbox, population)
        offspring = list(map(toolbox.clone, offspring))

        # Crossover and mutation (DEAP varAnd)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Find individuals with invalid fitness and evaluate
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # print(f"{len(invalid_ind)} offspring need fitness eval")
        eval_pop(invalid_ind, env)

        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the old population with the new offspring
        replacement(population, offspring)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if log_file:
            log(gen, ngen, population, stats, log_file, experiment_name)

    return population, logbook


def run_eaSimple(env, npop=50, ngen=10, cxpb=0.6, mutpb=0.2, experiment_name='dummy_demo_eaSimple'):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
        
    print(f"\nRUNNING EVOLUTION WITH eaSimple FOR ENEMY {env.enemies[0]}, EXPERIMENT {experiment_name}\n")
        
    toolbox = init_deap(env)
        
    population = init_pop(toolbox, npop)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    halloffame = tools.HallOfFame(1)

    # Initialize log 
    with open(os.path.join(experiment_name, 'results.txt'), 'a') as log_file:
        log_file.write("\ngen best mean std\n")
        start = time.time()

        population = eaSimple(population, toolbox, cxpb, mutpb, ngen, env,
                      stats=stats, halloffame=halloffame, log_file=log_file, experiment_name=experiment_name)


        end = time.time()  # End timer
        print(f"\nExecution time: {round((end - start) / 60, 2)} minutes \n")

    env.state_to_log()

if __name__ == "__main__":
    run_eaSimple()
