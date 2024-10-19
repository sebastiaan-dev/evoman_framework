from abc import ABC, abstractmethod


class DifficultyAdjuster(ABC):
    @abstractmethod
    def adjust_difficulty(self, generation, population):
        pass
