import os

from abc import ABC
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hebo import HEBOSearch
from ray.tune.search import ConcurrencyLimiter


class HyperOpt(ABC):
    def __init__(
        self, metric, mode, name, search_space, algo, num_samples, stopping_value
    ):
        scheduler = ASHAScheduler(
            # Metric to optimize
            metric=metric,
            # Maximize the fitness
            mode=mode,
            # Maximum number of generations/iterations (determined by tune.report calls)
            max_t=150,
            # Minimum number of generations before stopping if a trial is not performing well
            grace_period=20,
            # Reduction factor for number of trials
            reduction_factor=3,
        )

        # Use the HEBO search algorithm to find new hyperparameters based on the previous trials.
        # We use a concurrency limiter so the search algorithm can base its decisions on previous trials more efficiently.
        hebo = ConcurrencyLimiter(
            HEBOSearch(metric=metric, mode=mode), max_concurrent=4
        )

        tune_config = tune.TuneConfig(
            # Amount of trials to be run over the search space, higher values will result in better hyperparameters
            # but requires more resources.
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=hebo,
        )

        run_config = train.RunConfig(
            storage_path=f"{os.getcwd()}/ray_results",
            name=name,
            stop={metric: stopping_value},
        )

        self.tuner = tune.Tuner(
            algo,
            param_space=search_space,
            tune_config=tune_config,
            run_config=run_config,
        )

    def fit(self):
        self.tuner.fit()
