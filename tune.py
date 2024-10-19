import os

from ray import tune, train
from hyperopt.hyperopt import HyperOpt
from ea_algorithm.implementations.nsga3 import NSGA3


def tune_nsga3(config):

    nsga3 = NSGA3(
        ngen=120,
        enemies=[1, 2, 3],
        n_hidden=10,
        P=config["p"],
        cxpb=config["cxpb"],
        mutpb=config["mutpb"],
        mate_eta=config["mate_eta"],
        mutate_eta=config["mutate_eta"],
        train=train,
    )

    nsga3.evolve()


def optimize_nsga3():
    search_space = {
        "cxpb": tune.uniform(0.5, 1.0),
        "mutpb": tune.uniform(0.0, 0.6),
        "p": tune.randint(6, 30),
        "mate_eta": tune.randint(1, 30),
        "mutate_eta": tune.randint(1, 30),
    }

    tuner = HyperOpt(
        name="nsga3",
        metric="gain",
        mode="max",
        search_space=search_space,
        algo=tune_nsga3,
        num_samples=50,
    )

    tuner.fit()


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    # This is required as ray tune will otherwise change the working directory to the trial directory.
    # Which is at the root of the filesystem, which will make the Environment class unavailable.
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    optimize_nsga3()
