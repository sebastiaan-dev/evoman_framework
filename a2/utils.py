import math
import os

import numpy as np


def create_directory_structure(experiment_name, enemy_num, run_num=1):
    """
    Creates a directory structure for saving results based on the enemy and run number.
    Example: optimization_test_eaSimple_generalist/E1/run1/ etc.
    """
    enemy_dir = f"{experiment_name}/E{' '.join(str(e) for e in enemy_num)}"
    if not os.path.exists(enemy_dir):
        os.makedirs(enemy_dir)
    return enemy_dir


def fitness(individual, env):
    """
    Define a new fitness function which assigns a higher importantce to player health,
    take risks but progressively punish the agent more when losing excessive health.

    We could explore multiple fitness functions to check for differences in performance.
    """
    individual_np = np.array(individual)
    f, p, e, t = env.play(pcont=individual_np)

    fitness = 0.9 * (100 - e) - math.pow(math.e, (100 - p) / 40) - np.log(t)

    return (fitness,)
