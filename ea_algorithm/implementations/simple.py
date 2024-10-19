import numpy as np
import random
import multiprocessing as mp


from demo_controller import player_controller
from ea_algorithm.ea_algorithm import EAAlgorithm
from deap import creator, base, tools, algorithms
from evoman.environment import Environment


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

        return f

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


class Simple(EAAlgorithm):
    def __init__(
        self,
        ngen,
        npop,
        enemies,
        n_hidden,
        cxpb,
        mutpb,
        difficulty_adjuster=None,
        mutation_adjuster=None,
        train=None,
        name="tmp",
    ):
        super().__init__(
            base.Toolbox(),
            tools.Statistics(lambda ind: ind.fitness.values),
            n_hidden,
            enemies,
            difficulty_adjuster,
            mutation_adjuster,
            name,
        )

        self.train = train
        self.n_runs_per_enemy = 5
        self.ngen = ngen
        self.npop = npop
        self.cxpb = cxpb
        self.mutpb = mutpb

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
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        n_weights = (self.env.get_num_sensors() + 1) * self.n_hidden + (
            self.n_hidden + 1
        ) * 5

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
        self.toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
        self.toolbox.register(
            "mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2
        )  # Gaussian mutation
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate(self, individual):
        total_fitness = 0

        # Put tasks in the queue for each combination of enemy and how many times to run
        for enemy in self.enemies:
            for _ in range(self.n_runs_per_enemy):
                self.task_queue.put((individual, enemy))

        # Get results from the result queue
        for _ in range(len(self.enemies) * self.n_runs_per_enemy):
            fitness = self.result_queue.get()

            total_fitness += fitness

        total_runs = len(self.enemies) * self.n_runs_per_enemy
        avg_fitness = total_fitness / total_runs

        return (avg_fitness,)

    def create_offspring(self, population):
        offspring = self.toolbox.select(population, len(population))
        offspring = list(map(self.toolbox.clone, offspring))
        # Crossover and mutation (DEAP varAnd)
        return algorithms.varAnd(offspring, self.toolbox, self.cxpb, self.mutpb)

    def next_population(self, population, offspring):
        population[:] = offspring
        return population

    def cleanup(self):
        super().cleanup()
        self.stop_signal.set()

    def report(self, fitness, ind):
        if self.train is not None:
            value, _ = fitness

            self.train.report(
                {
                    "fitness": value,
                    "individual": ind,
                }
            )
