# imports framework
from math import factorial
import os
import random
from deap import creator, base, tools, algorithms
import numpy as np

from evoman.environment import Environment


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


def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


def evalOneMax(individual, env: Environment):
    n_runs_per_enemy = 3
    individual_np = np.array(individual)

    # total_fitness = 0
    total_player_health = 0
    total_enemy_health = 0
    # total_gain = 0
    total_time = 0

    list_of_enemies = env.enemies

    for enemy in list_of_enemies:
        env.update_parameter("enemies", [enemy])

        for _ in range(n_runs_per_enemy):
            f, p, e, t = env.play(pcont=individual_np, econt=enemy)

            # total_fitness += f
            total_player_health += p
            total_enemy_health += e
            # total_gain += p - e
            total_time += t

    env.enemies = list_of_enemies

    total_runs = len(env.enemies) * n_runs_per_enemy

    avg_player_health = total_player_health / total_runs
    avg_enemy_health = total_enemy_health / total_runs
    avg_time = total_time / total_runs

    # Normalize
    # gain_min, gain_max = -100 * total_runs, 100 * total_runs
    player_health_min, player_health_max = 0, 100
    enemy_health_min, enemy_health_max = 0, 100
    time_min, time_max = 0, env.timeexpire

    # normalized_gain = normalize(total_gain, gain_min, gain_max)
    normalized_player_health = normalize(
        avg_player_health, player_health_min, player_health_max
    )
    normalized_enemy_health = normalize(
        avg_enemy_health, enemy_health_min, enemy_health_max
    )
    normalized_time = normalize(avg_time, time_min, time_max)

    return (
        # normalized_gain,
        normalized_player_health,
        normalized_enemy_health,
        normalized_time,
    )


# initialize DEAP
def init_deap(
    env,
    P,
    mate_eta,
    mutate_eta,
):
    NOBJ = 3
    H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
    npop = int(H + (4 - H % 4))
    # NN paramtetrs
    n_hidden = 10
    n_weights = (env.get_num_sensors() + 1) * n_hidden + (n_hidden + 1) * 5
    ref_points = tools.uniform_reference_points(NOBJ, P)

    if not hasattr(creator, "FitnessMulti"):
        creator.create(
            "FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0)
        )  # Maximize player health, minimize enemy health and time
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

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
    toolbox.register(
        "mate", tools.cxSimulatedBinaryBounded, low=-1.0, up=1.0, eta=mate_eta
    )  # 30
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=-1.0,
        up=1.0,
        eta=mutate_eta,  # 20
        indpb=1.0 / n_weights,
    )
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    return toolbox, npop


# Step 1: Initialize population
def init_pop(toolbox, npop):
    return toolbox.population(n=npop)


# Step 2: Evaluate population using evaluation function (evalOneMax)
def eval_pop(individuals, toolbox, env):
    """Evaluate the fitness of a list of individuals."""
    fitnesses = list(map(lambda ind: toolbox.evaluate(ind, env), individuals))
    for ind, fit in zip(individuals, fitnesses):
        ind.fitness.values = fit


# Step 3: Select the next generation individuals
def selection(toolbox, population):
    return toolbox.select(population, len(population))


def run_nsga3_generalist(
    env,
    train,
    ngen,
    cxpb,
    mutpb,
    P,
    mate_eta,
    mutate_eta,
    experiment_name,
):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    toolbox, npop = init_deap(
        env,
        P,
        mate_eta,
        mutate_eta,
    )

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

        # Evaluate initial population
        eval_pop(population, toolbox, env)

        if halloffame is not None:
            halloffame.update(population)

        if log_file:
            log(0, ngen, population, stats, log_file, experiment_name)

        # Evolve
        for gen in range(1, ngen + 1):
            # Crossover and mutation (DEAP varAnd)
            offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)

            # Find individuals with invalid fitness and evaluate
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            eval_pop(invalid_ind, toolbox, env)

            if halloffame is not None:
                halloffame.update(offspring)

            combined_population = population + offspring
            population = toolbox.select(combined_population, npop)

            if log_file:
                log(gen, ngen, population, stats, log_file, experiment_name)

            best_ind = tools.selBest(population, 1)[0]
            best_player_health, best_enemy_health, best_time = best_ind.fitness.values

            train.report(
                {
                    "gain": best_player_health - best_enemy_health,
                    "player_health": best_player_health,
                    "enemy_health": best_enemy_health,
                    "time": best_time,
                    "individual": best_ind,
                }
            )

    env.state_to_log()

    # Save the weights of the best individual
    best_ind = tools.selBest(population, 1)[0]
    best_gain, best_player_health, best_enemy_health, best_time = (
        best_ind.fitness.values
    )

    save_best_dir = f"{experiment_name}"
    if not os.path.exists(save_best_dir):
        os.makedirs(save_best_dir)
    np.savetxt(
        os.path.join(
            save_best_dir,
            f"best-gain[{best_gain}]-ph[{best_player_health}]-eh[{best_enemy_health}]-t[{best_time}].txt",
        ),
        best_ind,
    )

    return found_max_fitness, halloffame[0]
