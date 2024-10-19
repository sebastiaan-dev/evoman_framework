import os

import numpy as np

from difficulty_adjuster.implementations.curriculum_learning import CurriculumLearning
from ea_algorithm.implementations.mupluslambda import MuPlusLambda
from ea_algorithm.implementations.nsga3 import NSGA3
from ea_algorithm.implementations.simple import Simple
from mutation_adjuster.implementations.aggressive import AggressiveAdjuster


def train_nsga3():
    aggressive = AggressiveAdjuster(cxpb=0.8931363224983215, mutpb=0.24)
    currlearning = CurriculumLearning([1, 2])
    nsga3 = NSGA3(
        ngen=2000,
        enemies=[1, 2],
        n_hidden=10,
        P=16,
        cxpb=0.8931363224983215,
        mutpb=0.24,
        mate_eta=5,
        mutate_eta=10,
        difficulty_adjuster=currlearning,
        mutation_adjuster=aggressive,
        name="nsga3-v5",
    )

    return nsga3.evolve()


def train_mupluslambda():
    muplus = MuPlusLambda(
        ngen=200,
        npop=150,
        enemies=[1, 2],
        n_hidden=10,
        cxpb=0.8,
        mutpb=0.14,
        lambda_=210,
        name="mupluslambda-v1",
    )

    return muplus.evolve()


def train_simple():
    simple = Simple(
        ngen=200,
        npop=150,
        enemies=[1, 2],
        n_hidden=10,
        cxpb=0.8,
        mutpb=0.14,
        name="simple-v1",
    )

    return simple.evolve()


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    train_simple()

    # np.savetxt()
