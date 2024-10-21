import os

import numpy as np

from difficulty_adjuster.implementations.curriculum_learning import CurriculumLearning
from ea_algorithm.implementations.mupluslambda import MuPlusLambda
from ea_algorithm.implementations.nsga3 import NSGA3
from ea_algorithm.implementations.simple import Simple
from mutation_adjuster.implementations.aggressive import AggressiveAdjuster


def train_nsga3(run, enemies):
    aggressive = AggressiveAdjuster(cxpb=0.7, mutpb=0.4)

    # Split the first 3 enemies into known and unknown enemies
    known_enemies = enemies[:1]
    unknown_enemies = enemies[1:]

    enemy_dir = "_".join(map(str, enemies))

    currlearning = CurriculumLearning(known_enemies, unknown_enemies)
    nsga3 = NSGA3(
        ngen=300,
        enemies=known_enemies,
        n_hidden=10,
        P=30,
        cxpb=0.9,
        mutpb=0.01,
        mate_eta=20,
        mutate_eta=30,
        difficulty_adjuster=currlearning,
        name=f"nsga3/{enemy_dir}",
        run=run,
    )

    return nsga3.evolve()


def train_mupluslambda(run, enemies):
    enemy_dir = "_".join(map(str, enemies))

    muplus = MuPlusLambda(
        ngen=50,
        npop=273,
        enemies=enemies,
        n_hidden=10,
        cxpb=0.8006317675051396,
        mutpb=0.05716553541228208,
        lambda_=227,
        name=f"mupluslambda/{enemy_dir}",
        run=run,
    )

    return muplus.evolve()


def train_simple(run, enemies):
    simple = Simple(
        ngen=60,
        npop=150,
        enemies=[1, 2, 3],
        n_hidden=10,
        cxpb=0.8,
        mutpb=0.14,
        name=f"simple-{'_'.join(map(str, enemies))}",
        run=run,
    )

    return simple.evolve()


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    runs = 10
    enemy_groups = [[1, 2, 3], [4, 7, 8]]

    for enemies in enemy_groups:
        for run in range(1, runs + 1):
            train_mupluslambda(run, enemies)
