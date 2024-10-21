import os

from ray import tune, train
from ea_algorithm.implementations.mupluslambda import MuPlusLambda
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
        should_checkpoint=False,
    )

    nsga3.evolve()


def optimize_nsga3():
    search_space = {
        "cxpb": tune.uniform(0.6, 0.9),
        "mutpb": tune.uniform(0.0, 0.5),
        "p": tune.randint(6, 35),
        "mate_eta": tune.randint(1, 30),
        "mutate_eta": tune.randint(1, 30),
    }

    tuner = HyperOpt(
        name="nsga3",
        metric="gain",
        mode="max",
        search_space=search_space,
        algo=tune_nsga3,
        num_samples=500,
        stopping_value=1.0,
    )

    best = tuner.fit()


def tune_mupluslambda(config):
    cxpb, mutpb = config["cxpb_mutpb"]

    muplus = MuPlusLambda(
        ngen=config["ngen"],
        npop=config["npop"],
        enemies=[1, 2, 3],
        n_hidden=10,
        cxpb=cxpb,
        mutpb=mutpb,
        lambda_=config["lambda_"],
        train=train,
        should_checkpoint=False,
    )

    muplus.evolve()


def sample_cxpb_mutpb():
    """
    Sample the crossover and mutation probabilities. This function allows for related hyperparameter sampling.

    """
    cxpb = tune.uniform(0.5, 0.9).sample()
    mutpb = tune.uniform(0.01, min(0.5, 1.0 - cxpb)).sample()
    return (cxpb, mutpb)


def optimize_mupluslambda():
    search_space = {
        "ngen": tune.randint(30, 70),
        "npop": tune.randint(50, 300),
        "cxpb_mutpb": tune.sample_from(sample_cxpb_mutpb),
        "lambda_": tune.randint(50, 300),
    }

    tuner = HyperOpt(
        name="mupluslambda",
        metric="fitness",
        mode="max",
        search_space=search_space,
        algo=tune_mupluslambda,
        num_samples=100,
        stopping_value=100,
    )

    tuner.fit()


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    # This is required as ray tune will otherwise change the working directory to the trial directory.
    # Which is at the root of the filesystem, which will make the Environment class unavailable.
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    optimize_mupluslambda()
