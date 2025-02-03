import numpy as np
from .base_selector import BasePoisonNodeSelector

class RandomSelector(BasePoisonNodeSelector):
    def select_poison_nodes(self, victim_nodes, num_poison_nodes):
        mapped_nodes = self._map_victim_nodes(victim_nodes)
        return list(np.random.choice(mapped_nodes, num_poison_nodes, replace=False))
