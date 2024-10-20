from abc import ABC, abstractmethod
from deap import base, tools

import os
import time as t
import numpy as np

from demo_controller import player_controller
from difficulty_adjuster.difficulty_adjuster import DifficultyAdjuster
from evoman.environment import Environment
from utils.math import normalize


class EAAlgorithm(ABC):
    def __init__(
        self,
        toolbox,
        stats,
        n_hidden,
        enemies,
        difficultyadjuster,
        mutationadjuster,
        name,
        should_checkpoint=True,
    ):
        self.difficultyadjuster = difficultyadjuster
        self.mutationadjuster = mutationadjuster
        self.toolbox = toolbox
        self.stats = stats
        self.enemies = enemies
        self.n_hidden = n_hidden
        self.name = name
        self.should_checkpoint = should_checkpoint

        self._setup_env()
        self.timeexpire = self.env.timeexpire
        self.toolbox = base.Toolbox()
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.halloffame = tools.HallOfFame(1)
        self._setup_stats()

    def _setup_env(self):
        self.env = Environment(
            randomini="yes",
            multiplemode="no",
            experiment_name="default",
            enemies=[1, 2, 3, 4, 5, 6, 7, 8],
            playermode="ai",
            player_controller=player_controller(self.n_hidden),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False,
            logs="off",
        )

    def _setup_stats(self):
        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    @abstractmethod
    def setup_deap(self):
        pass

    @abstractmethod
    def create_offspring(self, population):
        pass

    @abstractmethod
    def next_population(self, population, offspring):
        pass

    def report(self, gen, fitness, best_ind):
        pass

    def log(self, gen, population):
        pass

    def eval_pop(self, individuals):
        """Evaluate the fitness of a list of individuals."""
        fitnesses = list(map(self.toolbox.evaluate, individuals))
        for ind, fit in zip(individuals, fitnesses):
            ind.fitness.values = fit

    def adjust_difficulty(self, gen, population):
        if self.difficultyadjuster is not None:
            best_ind = best_ind = tools.selBest(population, 1)[0]
            new_enemies = self.difficultyadjuster.adjust_difficulty(
                gen, best_ind.fitness.values
            )
            if new_enemies is not None:
                self.enemies = new_enemies

                for ind in population:
                    del ind.fitness.values

                self.eval_pop(population)

    def adjust_mutation(self, gen, population):
        if self.mutationadjuster is not None:
            best_ind = best_ind = tools.selBest(population, 1)[0]
            result = self.mutationadjuster.adjust_mutation(gen, best_ind.fitness.values)
            if result is not None:
                new_mutpb, new_cxpb = result
                self.mutpb = new_mutpb
                self.cxpb = new_cxpb

    def cleanup(self):
        self.env.state_to_log()

    def update_halloffame(self, population):
        if self.halloffame is not None:
            self.halloffame.update(population)

    def checkpoint(self, gen, population):
        if not self.should_checkpoint:
            return

        # Save best individual to disk in folder self.name
        best_ind = tools.selBest(population, 1)[0]
        # Check if the folder exists, if not create it
        path = f"checkpoints/{self.name}"
        if not os.path.exists(path):
            os.makedirs(path)
        np.savetxt(f"{path}/{gen}-best.txt", best_ind)

    def evolve(self):
        population = self.toolbox.population(n=self.npop)

        # Evaluate initial population
        self.eval_pop(population)
        self.update_halloffame(population)
        best_ind = None

        self.log(0, population)

        # Evolve
        for gen in range(1, self.ngen + 1):
            self.adjust_mutation(gen, population)
            self.adjust_difficulty(gen, population)
            # Crossover and mutation (DEAP varAnd)
            offspring = self.create_offspring(population)

            # Find individuals with invalid fitness and evaluate
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            self.eval_pop(invalid_ind)

            self.update_halloffame(population)

            population = self.next_population(population, offspring)

            best_ind = best_ind = tools.selBest(population, 1)[0]
            self.report(gen, best_ind.fitness.values, best_ind)

            if gen % 10 == 0:
                self.checkpoint(gen, population)

            self.log(gen, population)

        self.checkpoint(gen, population)
        self.cleanup()

        return best_ind
