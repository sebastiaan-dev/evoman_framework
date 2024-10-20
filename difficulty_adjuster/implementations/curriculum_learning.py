import random
from difficulty_adjuster.difficulty_adjuster import DifficultyAdjuster
from utils.math import normalize


class CurriculumLearning(DifficultyAdjuster):
    def __init__(self, known_enemies, unkown_enemies):
        super().__init__()

        self.known_enemies = known_enemies
        self.unknown_enemies = unkown_enemies
        self.stagnation_threshold = 30
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

        self.previous_metric = current_metric
        return self.stagnation_counter >= self.stagnation_threshold

    def evaluate_metric(self, best_fitness):
        player_health, enemy_health, time = best_fitness

        return normalize(player_health - enemy_health, -1, 1)

    def adjust_difficulty(self, generation, best_fitness):
        metric = self.evaluate_metric(best_fitness)

        # Adjust difficulty dynamically, or change enemies based on generation number
        if self.detect_stagnation(metric) or metric == 1:
            new_enemies = []

            # If there are unkown enemies, select 1 unkown enemy and 1 known enemies at random
            if self.unknown_enemies:
                new_enemies.append(random.choice(self.unknown_enemies))
                new_enemies.extend(random.sample(self.known_enemies, 1))

                # Move the unknown enemy to known enemies
                self.known_enemies.append(new_enemies[0])
                self.unknown_enemies.remove(new_enemies[0])
            # If there are no uknown enemies, select known enemies at random
            else:
                if generation < 200:
                    new_enemies = random.sample(
                        self.known_enemies, random.randint(2, 3)
                    )
                else:
                    new_enemies = self.known_enemies

            self.stagnation_counter = 0

            print(
                f"[Gen {generation}] Stagnation detected, adjusting curriculum to {new_enemies}"
            )

            return new_enemies
        return None
