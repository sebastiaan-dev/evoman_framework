
################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
import random

from evoman.environment import Environment
from deap import creator, base, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt
from demo_controller import player_controller
experiment_name = 'dummy_demo'
n_hidden_neurons  = 10
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                    enemies=[7,8],
                    multiplemode="yes",
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False,
        )
n_hidden = 10
n_weights = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 
print("weighhhtss", n_weights)
def evalOneMax(individual):
        individual_np = np.array(individual)
        f, p, e, t = env.play(
            pcont=individual_np)
        return f,
 
def init_deap():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n = n_weights)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint) 
    #crossover
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox
toolbox = init_deap()

def run_evolution (npop = 100, offspring_size = n_weights, ngen=30):
    population = toolbox.population(n=npop)
    hof = tools.HallOfFame(1)  # To keep track of the best individual
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    algorithms.eaMuPlusLambda(population, 
                                toolbox, 
                                mu=30, 
                                lambda_=offspring_size, 
                                cxpb=0.6, 
                                mutpb=0.3, 
                                ngen=ngen, 
                                stats=stats, 
                                halloffame=hof)

    record = stats.compile(population)
    return hof, record
