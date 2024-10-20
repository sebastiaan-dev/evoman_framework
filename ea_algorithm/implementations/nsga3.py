from math import factorial
import os

import numpy as np
import random
import multiprocessing as mp


from demo_controller import player_controller
from ea_algorithm.ea_algorithm import EAAlgorithm
from deap import creator, base, tools, algorithms
from evoman.environment import Environment
from utils.math import normalize


class Worker(mp.Process):
    def __init__(self, enemies, n_hidden, task_queue, result_queue, stop_signal):
        super().__init__()
        self.enemies = enemies
        self.n_hidden = n_hidden
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_signal = stop_signal  # Shared event to signal stopping

    def _setup_env(self):
        # Initialize the environment only once per worker
        self.env = Environment(
            randomini="yes",
            multiplemode="no",
            experiment_name="test",
            enemies=[1, 2, 3, 4, 5, 6, 7, 8],
            playermode="ai",
            player_controller=player_controller(self.n_hidden),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False,
            logs="off",
        )

    def evaluate_enemy(self, individual, enemy):
        individual_np = np.array(individual)
        self.env.update_parameter("enemies", [enemy])
        f, p, e, t = self.env.play(pcont=individual_np)

        return p, e, t

    def run(self):
        self._setup_env()

        while not self.stop_signal.is_set():
            try:
                task = self.task_queue.get(timeout=1)  # Wait for a task or timeout
            except mp.queues.Empty:
                continue  # No task yet, continue waiting

            individual, enemy = task
            result = self.evaluate_enemy(individual, enemy)
            self.result_queue.put(result)


class NSGA3(EAAlgorithm):
    def __init__(
        self,
        ngen,
        enemies,
        n_hidden,
        P,
        cxpb,
        mutpb,
        mate_eta,
        mutate_eta,
        difficulty_adjuster=None,
        mutation_adjuster=None,
        train=None,
        name="tmp",
        should_checkpoint=True,
        run=None,
    ):
        super().__init__(
            base.Toolbox(),
            tools.Statistics(lambda ind: ind.fitness.values),
            n_hidden,
            enemies,
            difficulty_adjuster,
            mutation_adjuster,
            name,
            should_checkpoint,
        )

        self.train = train
        self.n_runs_per_enemy = 5
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.NOBJ = 3
        self.P = P
        self.mate_eta = mate_eta
        self.mutate_eta = mutate_eta
        self.run = run

        self.setup_deap()
        self._setup_mp()

    def _setup_mp(self):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.stop_signal = mp.Event()
        self.workers = [
            Worker(
                self.enemies,
                self.n_hidden,
                self.task_queue,
                self.result_queue,
                self.stop_signal,
            )
            for _ in range(mp.cpu_count())
        ]

        for worker in self.workers:
            worker.start()

    def setup_deap(self):
        H = factorial(self.NOBJ + self.P - 1) / (
            factorial(self.P) * factorial(self.NOBJ - 1)
        )
        self.npop = int(H + (4 - H % 4))
        n_weights = (self.env.get_num_sensors() + 1) * self.n_hidden + (
            self.n_hidden + 1
        ) * 5
        ref_points = tools.uniform_reference_points(self.NOBJ, self.P)

        if not hasattr(creator, "FitnessMulti"):
            creator.create(
                "FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0)
            )  # Maximize player health, minimize enemy health and time
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox.register("attr_float", random.uniform, -1, 1)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            n=n_weights,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register(
            "mate", tools.cxSimulatedBinaryBounded, low=-1.0, up=1.0, eta=self.mate_eta
        )  # 30
        self.toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            low=-1.0,
            up=1.0,
            eta=self.mutate_eta,  # 20
            indpb=4.0 / n_weights,
        )
        self.toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    def evaluate(self, individual):
        total_player_health = 0
        total_enemy_health = 0
        total_time = 0

        # Put tasks in the queue for each combination of enemy and how many times to run
        for enemy in self.enemies:
            for _ in range(self.n_runs_per_enemy):
                self.task_queue.put((individual, enemy))

        # Get results from the result queue
        for _ in range(len(self.enemies) * self.n_runs_per_enemy):
            player_health, enemy_health, time = self.result_queue.get()

            if player_health == 100 and enemy_health == 100:
                # This individual ended in a draw, most likely due to a timeout, assign worst possible fitness
                total_player_health += 0
                total_enemy_health += 100
                total_time += time
            else:
                total_player_health += player_health
                total_enemy_health += enemy_health
                total_time += time

        total_runs = len(self.enemies) * self.n_runs_per_enemy

        avg_player_health = total_player_health / total_runs
        avg_enemy_health = total_enemy_health / total_runs
        avg_time = total_time / total_runs

        # Normalize
        player_health_min, player_health_max = 0, 100
        enemy_health_min, enemy_health_max = 0, 100
        time_min, time_max = 0, self.timeexpire

        normalized_player_health = normalize(
            avg_player_health, player_health_min, player_health_max
        )
        normalized_enemy_health = normalize(
            avg_enemy_health, enemy_health_min, enemy_health_max
        )
        normalized_time = normalize(avg_time, time_min, time_max)

        return (
            normalized_player_health,
            normalized_enemy_health,
            normalized_time * 0.02,
        )

    def create_offspring(self, population):
        return algorithms.varAnd(population, self.toolbox, self.cxpb, self.mutpb)

    def next_population(self, population, offspring):
        combined_population = population + offspring
        return self.toolbox.select(combined_population, self.npop)

    def cleanup(self):
        super().cleanup()
        self.stop_signal.set()

    def report(self, gen, fitness, ind):
        if self.train is not None:
            best_player_health, best_enemy_health, best_time = fitness

            self.train.report(
                {
                    "gain": normalize(best_player_health - best_enemy_health, -1, 1),
                    "player_health": best_player_health,
                    "enemy_health": best_enemy_health,
                    "time": best_time,
                    "individual": ind,
                }
            )

    def log(self, gen, population):
        if not self.run:
            return
        # Check if the folder exists, if not create it
        path = f"results/{self.name}/{'_'.join(map(str, self.enemies))}/run{self.run}"

        # If the file does not exist, create it and write the header
        if not os.path.exists(path):
            os.makedirs(path)
            log_file = open(f"{path}/result.txt", "a")
            log_file.write("\ngen best mean std\n")
            log_file.close()
            return

        record = self.stats.compile(population)
        best_ind = tools.selBest(population, 1)[0]
        best_fitness = best_ind.fitness.values[0]

        gen_mean = record["avg"]
        gen_std = np.std([ind.fitness.values[0] for ind in population])

        log_file = open(f"{path}/result.txt", "a")
        log_file.write(f"\n{gen} {best_fitness:.6f} {gen_mean:.6f} {gen_std:.6f}\n")
        log_file.close()
