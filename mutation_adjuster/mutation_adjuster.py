from abc import ABC, abstractmethod


class MutationAdjuster(ABC):
    def __init__(self, mutpb, cxpb):
        self.mutpb = mutpb
        self.cxpb = cxpb

        self.adjusted_mutpb = mutpb
        self.adjusted_cxpb = cxpb

    @abstractmethod
    def adjust_mutation(self, generation, population):
        pass
