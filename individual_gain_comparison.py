import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evoman.environment import Environment
from demo_controller import player_controller

# Settings for evoman
os.environ["SDL_VIDEODRIVER"] = "dummy"

repetitions = 1
speed = "fastest"
fullscreen = False
sound = "off"
visuals = False

n_enemies = 3
n_hidden = 10

enemies = range(1, n_enemies + 1)


def load_best_solutions(base_path, algorithm):
    """
    Load the best solutions from the specified algorithm.

    Expects the following directory structure:

    base_path/<algorithm>/E<enemy_nr>/run<run_nr>/best.txt

    Returns a dictionary where keys are (enemy_nr, run_nr) tuples and values are the best solutions.
    """
    results = {}

    # Get the results path for the specified algorithm
    algorithm_path = os.path.join(base_path, algorithm)
    if not os.path.exists(algorithm_path):
        raise Exception(f"Path {algorithm_path} does not exist.")

    # Loop over all enemies
    for enemy_dir in os.listdir(algorithm_path):
        enemy_path = os.path.join(algorithm_path, enemy_dir)
        if not os.path.isdir(enemy_path):
            raise Exception(f"Incorrect enemy solution structure.")

        # Loop over all runs
        for run_dir in os.listdir(enemy_path):
            run_path = os.path.join(enemy_path, run_dir)
            if not os.path.isdir(run_path):
                raise Exception(f"Incorrect run solution structure.")

            # Load the best solution
            best_solution_path = os.path.join(run_path, "best.txt")
            if not os.path.exists(best_solution_path):
                raise Exception(f"Best solution not found at {best_solution_path}.")

            best_solution = np.loadtxt(best_solution_path)
            results[(int(enemy_dir[1:]), int(run_dir[3:]))] = best_solution

    return results


# Calculate the mean gain for a specific enemy and run
def calc_mean_gain(best_solution, enemy):
    # Initialize the environment
    env = Environment(
        experiment_name="test",
        enemies=[enemy],
        playermode="ai",
        fullscreen=fullscreen,
        player_controller=player_controller(n_hidden),
        enemymode="static",
        level=2,
        sound=sound,
        speed=speed,
        visuals=visuals,
    )

    # Calculate the gain for each repetition
    gains = []
    for _ in range(repetitions):
        f, p, e, t = env.play(pcont=best_solution)
        gains.append(p - e)

    return np.mean(gains)


# Calculate the mean gain for each enemy and run
def calc_mean_gains(best_solutions):
    results = {}
    for (enemy, run), best_solution in best_solutions.items():
        results[(enemy, run)] = calc_mean_gain(best_solution, enemy)
    return results


# Calculate the mean gains for both algorithms
def calc_mean_gains_for_algorithms():
    # Load the best solutions for both algorithms
    eaSimple_best_solutions = load_best_solutions("", "runs_eaSimple")
    eaLambda_best_solutions = load_best_solutions("", "runs_muPlusLambda")
    eaLambda_gains = calc_mean_gains(eaLambda_best_solutions)
    eaSimple_gains = calc_mean_gains(eaSimple_best_solutions)

    return eaSimple_gains, eaLambda_gains


aSimple_gains, eaLambda_gains = calc_mean_gains_for_algorithms()

for enemy in enemies:
    # Filter the gains for the specific enemy
    eaSimple_enemy_gains = [
        gain for (enemy_nr, run_nr), gain in aSimple_gains.items() if enemy_nr == enemy
    ]
    eaLambda_enemy_gains = [
        gain for (enemy_nr, run_nr), gain in eaLambda_gains.items() if enemy_nr == enemy
    ]

    fig, ax = plt.subplots()
    ax.boxplot(
        [eaSimple_enemy_gains, eaLambda_enemy_gains], labels=["eaSimple", "eaLambda"]
    )
    ax.set_title(f"Enemy {enemy}")
    ax.set_ylabel("Mean gain")
    plt.show()
