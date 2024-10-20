import os

import numpy as np

from difficulty_adjuster.implementations.curriculum_learning import CurriculumLearning
from ea_algorithm.implementations.mupluslambda import MuPlusLambda
from ea_algorithm.implementations.nsga3 import NSGA3
from ea_algorithm.implementations.simple import Simple
from mutation_adjuster.implementations.aggressive import AggressiveAdjuster


def train_nsga3(run, enemies):
    aggressive = AggressiveAdjuster(cxpb=0.8931363224983215, mutpb=0.24)
    currlearning = CurriculumLearning(enemies)
    nsga3 = NSGA3(
        ngen=2000,
        enemies=enemies,
        n_hidden=10,
        P=16,
        cxpb=0.8931363224983215,
        mutpb=0.24,
        mate_eta=5,
        mutate_eta=10,
        difficulty_adjuster=currlearning,
        mutation_adjuster=aggressive,
        name=f"nsga3-{'_'.join(map(str, enemies))}",
        run=run,
    )

    return nsga3.evolve()


# {
#   "cxpb_mutpb": [
#     0.8006317675051396,
#     0.05716553541228208
#   ],
#   "lambda_": 227,
#   "ngen": 48,
#   "npop": 273
# }


def train_mupluslambda(run, enemies):
    muplus = MuPlusLambda(
        ngen=100,
        npop=273,
        enemies=enemies,
        n_hidden=10,
        cxpb=0.8006317675051396,
        mutpb=0.05716553541228208,
        lambda_=227,
        name=f"mupluslambda-{'_'.join(map(str, enemies))}",
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
        for run in range(runs):
            train_mupluslambda(run, enemies)
