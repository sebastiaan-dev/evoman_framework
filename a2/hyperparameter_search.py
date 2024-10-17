import os

from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search import ConcurrencyLimiter


from a2.simulate_ea import (
    simulate_ea_mupluslambda,
    simulate_ea_simple,
    simulate_ea_nsga3,
)

ea_algorithm = "nsga3"  # "simple" or "muPlusLambda" or "nsga3"


def train_ea(config):
    """
    Wrapper function for ray tune to train the EA with the given configuration.
    """
    cxpb, mutpb = config["cxpb_mutpb"]

    if ea_algorithm == "simple":
        args = (
            train,
            f"a2/{ea_algorithm}",
            [1, 4],
            10,
            config["population_size"],
            config["num_generations"],
            cxpb,
            mutpb,
        )

        fitness, ind = simulate_ea_simple(args)

        # Return the fitness and individual after finishing the last evolution.
        return {"fitness": fitness, "individual": ind}
    else:
        args = (
            train,
            f"a2/{ea_algorithm}",
            [1, 4],
            10,
            config["population_size"],
            config["num_generations"],
            cxpb,
            config["lambda_"],
            mutpb,
        )

        fitness, ind = simulate_ea_mupluslambda(args)

        # Return the fitness and individual after finishing the last evolution.
        return {"fitness": fitness, "individual": ind}


def sample_cxpb_mutpb():
    """
    Sample the crossover and mutation probabilities. This function allows for related hyperparameter sampling.

    """
    cxpb = tune.uniform(0, 1).sample()
    mutpb = tune.uniform(0, 1 - cxpb).sample()
    return (cxpb, mutpb)


def optimize_hyperparameters():
    """
    Optimize the hyperparameters of the EA using Ray Tune.
    """

    if ea_algorithm == "simple":
        search_space = {
            "population_size": tune.randint(60, 250),
            "num_generations": tune.randint(40, 70),
            "cxpb_mutpb": tune.sample_from(sample_cxpb_mutpb),
        }
    else:
        search_space = {
            "population_size": tune.randint(60, 250),
            "num_generations": tune.randint(40, 70),
            "cxpb_mutpb": tune.sample_from(sample_cxpb_mutpb),
            "lambda_": tune.randint(60, 250),
        }

    scheduler = ASHAScheduler(
        # Metric to optimize
        metric="fitness",
        # Maximize the fitness
        mode="max",
        # Maximum number of generations/iterations (determined by tune.report calls)
        max_t=100,
        # Minimum number of generations before stopping if a trial is not performing well
        grace_period=20,
        # Reduction factor for number of trials
        reduction_factor=2,
    )

    # Use the HEBO search algorithm to find new hyperparameters based on the previous trials.
    # We use a concurrency limiter so the search algorithm can base its decisions on previous trials more efficiently.
    hebo = ConcurrencyLimiter(
        HEBOSearch(metric="fitness", mode="max"), max_concurrent=5
    )

    tune_config = tune.TuneConfig(
        # Amount of trials to be run over the search space, higher values will result in better hyperparameters
        # but requires more resources.
        num_samples=50,
        scheduler=scheduler,
        search_alg=hebo,
    )

    run_config = train.RunConfig(
        storage_path=f"{os.getcwd()}/a2/ray_results",
        name=ea_algorithm,
        stop={"fitness": 90},
    )

    tuner = tune.Tuner(
        train_ea,
        param_space=search_space,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()

    # print("Best hyperparameters found were: ", analysis.best_config)


def train_ea_nsga3(config):
    """
    Wrapper function for ray tune to train the EA with the given configuration.
    """

    args = (
        train,
        f"a2/{ea_algorithm}",
        [1, 4, 6],
        10,
        config["num_generations"],
        config["cxpb"],
        config["mutpb"],
        config["p"],
        config["mate_eta"],
        config["mutate_eta"],
    )

    fitness, ind = simulate_ea_nsga3(args)

    # Return the fitness and individual after finishing the last evolution.
    # return {"gain": fitness, "individual": ind}


def optimize_hyperparameters_nsga3():
    """
    Optimize the hyperparameters of the EA using Ray Tune.
    """

    search_space = {
        "num_generations": tune.randint(30, 150),
        "cxpb": tune.uniform(0.5, 1.0),
        "mutpb": tune.uniform(0.0, 0.5),
        "p": tune.randint(10, 30),
        "mate_eta": tune.randint(2, 30),
        "mutate_eta": tune.randint(2, 30),
    }

    scheduler = ASHAScheduler(
        # Metric to optimize
        metric="gain",
        # Maximize the fitness
        mode="max",
        # Maximum number of generations/iterations (determined by tune.report calls)
        max_t=150,
        # Minimum number of generations before stopping if a trial is not performing well
        grace_period=20,
        # Reduction factor for number of trials
        reduction_factor=2,
    )

    # Use the HEBO search algorithm to find new hyperparameters based on the previous trials.
    # We use a concurrency limiter so the search algorithm can base its decisions on previous trials more efficiently.
    hebo = ConcurrencyLimiter(HEBOSearch(metric="gain", mode="max"), max_concurrent=8)

    tune_config = tune.TuneConfig(
        # Amount of trials to be run over the search space, higher values will result in better hyperparameters
        # but requires more resources.
        num_samples=250,
        scheduler=scheduler,
        search_alg=hebo,
    )

    run_config = train.RunConfig(
        storage_path=f"{os.getcwd()}/a2/ray_results",
        name=ea_algorithm,
        stop={"gain": 1.0},
    )

    tuner = tune.Tuner(
        train_ea_nsga3,
        param_space=search_space,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()


if __name__ == "__main__":
    # This is required as ray tune will otherwise change the working directory to the trial directory.
    # Which is at the root of the filesystem, which will make the Environment class unavailable.
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

    optimize_hyperparameters_nsga3()
