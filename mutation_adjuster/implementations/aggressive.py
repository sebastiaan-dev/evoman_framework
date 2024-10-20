from mutation_adjuster.mutation_adjuster import MutationAdjuster
from utils.math import normalize


class AggressiveAdjuster(MutationAdjuster):
    def __init__(self, mutpb, cxpb):
        super().__init__(mutpb, cxpb)

        self.stagnation_threshold = 2
        self.stagnation_counter = 0
        self.previous_metric = None

    def detect_stagnation(self, current_metric):
        if self.previous_metric is None:
            self.previous_metric = current_metric
            return False

        if abs(current_metric - self.previous_metric) < 1e-3:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.adjusted_mutpb = self.mutpb

        self.previous_metric = current_metric
        return self.stagnation_counter >= self.stagnation_threshold

    def evaluate_metric(self, best_fitness):
        player_health, enemy_health, time = best_fitness

        return normalize(player_health - enemy_health, -1, 1)

    def adjust_mutation(self, generation, best_fitness):
        metric = self.evaluate_metric(best_fitness)

        if self.detect_stagnation(metric):
            self.adjusted_mutpb = min(0.5, self.adjusted_mutpb + 0.05)

            self.stagnation_counter = 0

            print(f"[Gen {generation}] Adjusting mutpb to {self.adjusted_mutpb}")

            return self.adjusted_mutpb, self.cxpb
        return None
