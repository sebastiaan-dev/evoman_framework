import os
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


class MuPlusLambda(EAAlgorithm):
    def __init__(
        self,
        ngen,
        npop,
        enemies,
        n_hidden,
        cxpb,
        mutpb,
        lambda_,
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
        self.npop = npop
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.lambda_ = lambda_
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
        return algorithms.varOr(
            population, self.toolbox, self.lambda_, self.cxpb, self.mutpb
        )

    def next_population(self, population, offspring):
        combined_population = population + offspring
        return self.toolbox.select(combined_population, self.npop)

    def cleanup(self):
        super().cleanup()
        self.stop_signal.set()

    def report(self, gen, fitness, ind):
        print(f"Generation {gen}: {fitness}")

        if self.train is not None:
            self.train.report(
                {
                    "fitness": fitness[0],
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
