################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
################################

# imports framework
import sys, os
import random
import time
from evoman.environment import Environment
from deap import creator, base, tools, algorithms
import numpy as np
from demo_controller import player_controller

experiment_name = 'dummy_demo_muPlusLambda'
n_hidden_neurons = 10
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

#log for visualization
def log(gen, population, stats, log_file):
    record = stats.compile(population)
    best_ind = tools.selBest(population, 1)[0]
    best_fitness = best_ind.fitness.values[0]

    gen_mean = record['avg']
    gen_std = np.std([ind.fitness.values[0] for ind in population])
    print(f"\n GENERATION {gen}  {best_fitness:.6f} {gen_mean:.6f} {gen_std:.6f}")
    log_file.write(f"\n{gen} {best_fitness:.6f} {gen_mean:.6f} {gen_std:.6f}\n")
    np.savetxt(experiment_name + '/best.txt', best_ind)

#initialize environment
env = Environment(experiment_name=experiment_name,
                  enemies=[7, 8],
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

#NN parameters
n_hidden = 10
n_weights = (env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5

#fitness evaluation function
def evalOneMax(individual):
    individual_np = np.array(individual)
    f, p, e, t = env.play(pcont=individual_np)
    return f,

#initialize DEAP
def init_deap():
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except AttributeError:
        pass
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_weights)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)  #two-point crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  #gaussian mutation
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

toolbox = init_deap()
#initiliazes population using deap population function which calls registered individual function repedately.
def init_pop(mu):
    return toolbox.population(n=mu)

#evaluate population using registered evaluate function (evalOneMax)
def eval_pop(population):
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

#select best indivuduals among combined population and offspring
def selection(population, offspring, mu):
    return toolbox.select(population + offspring, mu)

#crossover using two-point crossover with probability cxpb
def crossover(offspring, cxpb):
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

#mutation
def mutate(offspring, mutpb):
    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

#replacement to form new population
def replace_pop(parents, offspring, mu):
    return selection(parents, offspring, mu)

'''
Each step of evalutionary algorithm is implemented as seperate functions. Order of steps:
1- init_pop
2- eval_pop 
3- selection
4- crossover
5- mutation
6- replacement
'''

def run_evolution(mu=100, ngen=30, lambda_=100, cxpb=0.6, mutpb=0.2):
    population = init_pop(mu)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    halloffame = tools.HallOfFame(1)
    
    #init log
    log_file = open(experiment_name + '/results.txt', 'a')
    log_file.write("\ngen best mean std\n")
    start = time.time()
    
    # Step 2: Evaluate the initial population
    eval_pop(population)
    halloffame.update(population)
    log(0, population, stats, log_file)
    
    # Evolve
    for gen in range(1, ngen + 1):
        # Generate offspring (crossover and mutation handled by varOr())
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # crossover(offspring, cxpb)
        # mutate(offspring, mutpb)

        # Evaluate offspring
        eval_pop(offspring)

        # Replacement: (select the next generation from parents + offspring)
        population = replace_pop(population, offspring, mu)
        halloffame.update(population)

        log(gen, population, stats, log_file)
    
    end = time.time()  # End timer
    print(f"\nExecution time: {round((end - start) / 60, 2)} minutes \n")
    log_file.close()
    env.state_to_log()


if __name__ == "__main__":
    run_evolution()